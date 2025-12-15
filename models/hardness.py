import torch
from torch import nn
import torch.nn.functional as F


class HardnessEstimator(nn.Module):
    """
    只做“难度估计”，不参与反向传播。
    对每个样本得到一个 hardness ∈ [0,1]，数值越大表示越“难”。

    hardness = alpha_pred * h_pred + alpha_rec * h_rec

    - h_pred：基于预测误差 (prediction error)
    - h_rec：基于重构误差 (reconstruction error)
    """
    def __init__(self,
                 train_mode: str = 'regression',
                 alpha_pred: float = 1.0,
                 alpha_rec: float = 0.3,
                 eps: float = 1e-6):
        super().__init__()
        self.train_mode = train_mode
        self.alpha_pred = float(alpha_pred)
        self.alpha_rec = float(alpha_rec)
        self.eps = float(eps)

    @staticmethod
    def _min_max_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
        """
        在一个 batch 内做 min-max 归一化：x_norm = (x - min) / (max - min + eps)
        若 max==min，则返回全 0，避免除 0。
        """
        eps_val = float(eps)

        x_min = x.min()
        x_max = x.max()
        denom = (x_max - x_min).abs().item()  # 转成标量再比较，避免 Tensor 与 float 比较出错

        if denom < eps_val:
            return torch.zeros_like(x)

        return (x - x_min) / ((x_max - x_min) + eps_val)

    def _pred_error(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        预测误差：
          - 回归：逐样本 MSE
          - 分类：1 - p_correct
        返回 shape: [B]
        """
        preds_flat = preds.view(preds.size(0), -1)
        labels_flat = labels.view(labels.size(0), -1)

        if self.train_mode == 'regression':
            err = (preds_flat - labels_flat) ** 2  # [B, D_out]
            err = err.mean(dim=1)                  # [B]
        else:
            labels_long = labels_flat.view(-1).long()
            prob = F.softmax(preds_flat, dim=-1)   # [B, C]
            p_correct = prob.gather(dim=-1, index=labels_long.unsqueeze(-1)).squeeze(-1)
            err = 1.0 - p_correct                  # [B]

        return err

    def _rec_error(self,
                   rec_feats: torch.Tensor,
                   complete_feats: torch.Tensor) -> torch.Tensor:
        """
        重构误差：逐样本 MSE
        rec_feats 和 complete_feats 是 P-RMF 中拼接的 [B, *] 特征
        """
        if rec_feats is None or complete_feats is None:
            return None

        B = rec_feats.size(0)
        rec_flat = rec_feats.view(B, -1)
        comp_flat = complete_feats.view(B, -1)

        err = (rec_flat - comp_flat) ** 2  # [B, D_total]
        err = err.mean(dim=1)              # [B]
        return err

    def forward(self,
                preds: torch.Tensor,
                labels: torch.Tensor,
                rec_feats: torch.Tensor = None,
                complete_feats: torch.Tensor = None) -> torch.Tensor:
        """
        输入：
          - preds:           [B, 1] 或 [B, C]，来自 out['sentiment_preds']
          - labels:          [B] 或 [B,1]，来自 label['sentiment_labels']
          - rec_feats:       [B, ...]，来自 out['rec_feats']
          - complete_feats:  [B, ...]，来自 out['complete_feats']

        输出：
          - hardness: [B]，每个样本一个难度分数，范围 [0,1]
        """
        with torch.no_grad():
            # 1) 预测误差 → h_pred
            e_pred = self._pred_error(preds, labels)     # [B]
            h_pred = self._min_max_norm(e_pred, self.eps)

            # 2) 重构误差 → h_rec（若有完整模态）
            h_rec = None
            if (rec_feats is not None) and (complete_feats is not None):
                e_rec = self._rec_error(rec_feats, complete_feats)  # [B]
                h_rec = self._min_max_norm(e_rec, self.eps)

            # 3) 组合为最终 hardness
            hardness = self.alpha_pred * h_pred
            if (h_rec is not None) and (self.alpha_rec > 0.0):
                hardness = hardness + self.alpha_rec * h_rec

            hardness = torch.clamp(hardness, 0.0, 1.0)

        return hardness
