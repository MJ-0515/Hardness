from torch import nn
from torch.nn import functional as F
import torch

class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.kl = args['base']['kl']
        self.CE_Fn = nn.CrossEntropyLoss()
        self.MSE_Fn = nn.MSELoss()
        self.L1 = nn.L1Loss()

    def forward(self, out, label, sample_weights=None):
        """
        sample_weights: [B] 或 None
          - 为 None 时，使用标准 MSE
          - 不为 None 时，对情感预测误差做样本级加权
        """
        rec_feats = out['rec_feats']
        complete_feats = out['complete_feats']

        # 重构 loss：仍然是整体 MSE
        if (rec_feats is not None) and (complete_feats is not None):
            l_rec = self.MSE_Fn(rec_feats, complete_feats)
        else:
            l_rec = 0.0

        preds = out['sentiment_preds']
        targets = label['sentiment_labels']

        # 情感预测 loss：支持样本级加权
        if sample_weights is not None:
            # [B,*] -> [B, D_out]
            preds_flat = preds.view(preds.size(0), -1)
            targets_flat = targets.view(targets.size(0), -1)

            # 逐样本 MSE
            per_sample_mse = ((preds_flat - targets_flat) ** 2).mean(dim=1)  # [B]

            w = sample_weights.view(-1)
            # 归一化权重，使得平均权重为 1，不改变整体 loss 尺度
            w = w / (w.mean() + 1e-8)

            l_sp = (w * per_sample_mse).mean()
        else:
            l_sp = self.MSE_Fn(preds, targets)

        l_kl = out["kl_loss"]
        loss = self.gamma * l_rec + self.sigma * l_sp + self.kl * l_kl

        return {'loss': loss, 'l_sp': l_sp, 'l_rec': l_rec, 'l_kl': l_kl}
