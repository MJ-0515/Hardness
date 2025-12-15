import torch
from torch import nn
import torch.nn.functional as F

class HardnessEstimator(nn.Module):
    def __init__(self,
                 train_mode: str = 'regression',
                 alpha_pred: float = 1.0,
                 alpha_rec: float = 0.3,
                 eps: float = 1e-6,
                 momentum: float = 0.99): # 新增动量参数
        super().__init__()
        self.train_mode = train_mode
        self.alpha_pred = float(alpha_pred)
        self.alpha_rec = float(alpha_rec)
        self.eps = float(eps)
        
        # 注册缓冲区来保存运行时的均值，不参与梯度传播，但随模型保存
        self.register_buffer('running_pred_err', torch.tensor(1.0))
        self.register_buffer('running_rec_err', torch.tensor(1.0))
        self.momentum = momentum

    def _update_running_stats(self, current_mean, running_mean):
        """使用动量更新运行时的误差均值"""
        if running_mean == 1.0: # 初始化状态
            return current_mean.detach()
        return self.momentum * running_mean + (1 - self.momentum) * current_mean.detach()

    def _dynamic_tanh_norm(self, err: torch.Tensor, running_mean: torch.Tensor) -> torch.Tensor:
        """
        动态 Tanh 归一化：
        使用运行均值作为基准。
        当 err == running_mean 时，输出 tanh(1.0) ≈ 0.76
        当 err 很大时，趋向 1.0；很小时趋向 0.0
        """
        # 避免除以 0
        scale = running_mean + self.eps
        return torch.tanh(err / scale)

    def _pred_error(self, preds, labels):
        preds_flat = preds.view(preds.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        if self.train_mode == 'regression':
            return ((preds_flat - labels_flat) ** 2).mean(dim=1)
        else:
            # Classification logic...
            labels_long = labels_flat.view(-1).long()
            prob = F.softmax(preds_flat, dim=-1)
            p_correct = prob.gather(dim=-1, index=labels_long.unsqueeze(-1)).squeeze(-1)
            return 1.0 - p_correct

    def _rec_error(self, rec_feats, complete_feats):
        B = rec_feats.size(0)
        rec_flat = rec_feats.view(B, -1)
        comp_flat = complete_feats.view(B, -1)
        return ((rec_flat - comp_flat) ** 2).mean(dim=1)

    def forward(self, preds, labels, rec_feats=None, complete_feats=None):
        with torch.no_grad():
            # 1) 计算原始预测误差
            e_pred = self._pred_error(preds, labels) # [B]
            
            # 更新运行均值 (训练阶段)
            if self.training:
                self.running_pred_err = self._update_running_stats(e_pred.mean(), self.running_pred_err)
            
            # 动态归一化
            h_pred = self._dynamic_tanh_norm(e_pred, self.running_pred_err)

            # 2) 计算原始重构误差
            h_rec = None
            if (rec_feats is not None) and (complete_feats is not None):
                e_rec = self._rec_error(rec_feats, complete_feats)
                
                if self.training:
                    self.running_rec_err = self._update_running_stats(e_rec.mean(), self.running_rec_err)
                
                h_rec = self._dynamic_tanh_norm(e_rec, self.running_rec_err)

            # 3) 组合
            hardness = self.alpha_pred * h_pred
            if (h_rec is not None) and (self.alpha_rec > 0.0):
                hardness = hardness + self.alpha_rec * h_rec

            # 截断
            hardness = torch.clamp(hardness, 0.0, 1.0)

        return hardness
