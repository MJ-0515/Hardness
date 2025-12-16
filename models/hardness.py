import torch
from torch import nn
import torch.nn.functional as F

class HardnessEstimator(nn.Module):
    def __init__(self,
                 train_mode: str = 'regression',
                 alpha_pred: float = 1.0,
                 alpha_rec: float = 0.3,
                 eps: float = 1e-6,
                 momentum: float = 0.99,
                 temp: float = 2.0):  # [新增] 温度系数，默认2.0
        super().__init__()
        self.train_mode = train_mode
        self.alpha_pred = float(alpha_pred)
        self.alpha_rec = float(alpha_rec)
        self.eps = float(eps)
        self.temp = float(temp)
        
        # 注册缓冲区来保存运行时的均值
        self.register_buffer('running_pred_err', torch.tensor(1.0))
        self.register_buffer('running_rec_err', torch.tensor(1.0))
        self.momentum = momentum

    def _update_running_stats(self, current_mean, running_mean):
        """使用动量更新运行时的误差均值"""
        # 使用 .item() 确保标量比较的安全性
        if running_mean.item() == 1.0: 
            return current_mean.detach()
        # 训练时更新，评估时保持
        if self.training:
            return self.momentum * running_mean + (1 - self.momentum) * current_mean.detach()
        return running_mean

    def _dynamic_tanh_norm(self, err: torch.Tensor, running_mean: torch.Tensor) -> torch.Tensor:
        """
        动态 Tanh 归一化：
        引入温度系数 temp。
        当 temp=2.0 时，err == running_mean -> tanh(0.5) ≈ 0.46
        这让平均难度的样本 hardness 处于 0.5 附近，而不是 0.76，留出了更多上升空间。
        """
        scale = (running_mean * self.temp) + self.eps
        return torch.tanh(err / scale)

    # _pred_error 和 _rec_error 保持不变...
    def _pred_error(self, preds, labels):
        preds_flat = preds.view(preds.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)
        if self.train_mode == 'regression':
            return ((preds_flat - labels_flat) ** 2).mean(dim=1)
        else:
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
            # 1) 预测误差
            e_pred = self._pred_error(preds, labels)
            self.running_pred_err = self._update_running_stats(e_pred.mean(), self.running_pred_err)
            h_pred = self._dynamic_tanh_norm(e_pred, self.running_pred_err)

            # 2) 重构误差
            h_rec = None
            if (rec_feats is not None) and (complete_feats is not None):
                e_rec = self._rec_error(rec_feats, complete_feats)
                self.running_rec_err = self._update_running_stats(e_rec.mean(), self.running_rec_err)
                h_rec = self._dynamic_tanh_norm(e_rec, self.running_rec_err)

            # 3) 组合：改为凸组合 (加权平均)，防止溢出
            # hardness = (w1*h1 + w2*h2) / (w1+w2)
            total_alpha = self.alpha_pred
            weighted_sum = self.alpha_pred * h_pred
            
            if (h_rec is not None) and (self.alpha_rec > 0.0):
                weighted_sum += self.alpha_rec * h_rec
                total_alpha += self.alpha_rec
            
            hardness = weighted_sum / (total_alpha + 1e-8)
            
            # 虽然理论上在[0,1]，但加上clamp更安全
            hardness = torch.clamp(hardness, 0.0, 1.0)

        return hardness
