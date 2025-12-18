from torch import nn
import torch


class MultimodalLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gamma = args['base']['gamma']
        self.sigma = args['base']['sigma']
        self.kl = args['base']['kl']
        self.MSE = nn.MSELoss(reduction="none")

    def forward(self, out, label, sample_weight_pred=None, sample_weight_rec=None,
            sample_mask_pred=None, sample_mask_rec=None):

        pred = out['sentiment_preds'].view(-1)
        y = label['sentiment_labels'].view(-1)
        l_sp_vec = (pred - y).pow(2)

        rec = out.get('rec_feats', None)
        comp = out.get('complete_feats', None)
        if rec is not None and comp is not None:
          l_rec_vec = (rec - comp).pow(2).mean(dim=(1, 2))
        else:
          l_rec_vec = torch.zeros_like(l_sp_vec)

        if sample_mask_pred is not None:
           l_sp_vec = l_sp_vec * sample_mask_pred
        if sample_mask_rec is not None:
           l_rec_vec = l_rec_vec * sample_mask_rec

        if sample_weight_pred is not None:
           l_sp_vec = l_sp_vec * sample_weight_pred
           denom_sp = (sample_weight_pred * (sample_mask_pred if sample_mask_pred is not None else 1.0)).sum() + 1e-6
        else:
            denom_sp = (sample_mask_pred.sum() + 1e-6) if sample_mask_pred is not None else float(l_sp_vec.numel())

        if sample_weight_rec is not None:
            l_rec_vec = l_rec_vec * sample_weight_rec
            denom_rec = (sample_weight_rec * (sample_mask_rec if sample_mask_rec is not None else 1.0)).sum() + 1e-6
        else:
            denom_rec = (sample_mask_rec.sum() + 1e-6) if sample_mask_rec is not None else float(l_rec_vec.numel())

        l_sp = l_sp_vec.sum() / denom_sp
        l_rec = l_rec_vec.sum() / denom_rec

        loss = self.sigma * l_sp + self.gamma * l_rec + self.kl * out["kl_loss"]
        return {'loss': loss, 'l_sp': l_sp.detach(), 'l_rec': l_rec.detach(), 'l_kl': out["kl_loss"].detach()}
