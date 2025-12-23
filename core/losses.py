from torch import nn
import torch


class MultimodalLoss(nn.Module):
    """
    P-RMF 模型的综合损失函数模块。
    
    组成部分：
    1. 情感预测损失 (L_sp): 主任务 MSE Loss。
    2. 特征重构损失 (L_rec): 辅助任务 MSE Loss，用于保证模态表征的完整性。
    3. KL 散度损失 (L_kl): 用于变分信息瓶颈 (VIB) 或分布对齐的正则化项。
    
    特性：
    - 支持 Instance-wise Weighted Loss (样本级加权损失)。
    - 能够配合 Curriculum Learning，根据传入的 weights 或 masks 动态调整每个样本对梯度的贡献。
    """

    def __init__(self, args):
        super().__init__()
        self.gamma = args['base']['gamma'] # 重构损失系数
        self.sigma = args['base']['sigma'] # 预测损失系数
        self.kl = args['base']['kl']       # KL 散度损失系数

        # 关键点：reduction="none"
        # PyTorch 默认是 "mean"，会直接返回标量。
        # 这里必须设为 "none"，保留 (Batch_Size,) 形状的 Loss 向量，以便后续乘以样本权重。
        self.MSE = nn.MSELoss(reduction="none")

    def forward(self, out, label, sample_weight_pred=None, sample_weight_rec=None,
            sample_mask_pred=None, sample_mask_rec=None):
        
        """
        Args:
            out: 模型输出字典，包含 'sentiment_preds', 'rec_feats', 'kl_loss' 等。
            label: 标签字典，包含 'sentiment_labels'。
            sample_weight_pred: (B,) 预测任务的样本权重，由 Scheduler 计算得出 (包含 Gate 和 Reweight)。
            sample_weight_rec: (B,) 重构任务的样本权重。
            sample_mask_pred: (B,) 预测任务的硬门控 Mask (0/1)，仅在未提供 weight 时作为替补。
            sample_mask_rec: (B,) 重构任务的硬门控 Mask。
        """
        # ==========================
        # 1. 计算原始的样本级损失向量
        # ==========================
        
        # --- 主任务：情感预测 Loss ---
        pred = out['sentiment_preds'].view(-1)
        y = label['sentiment_labels'].view(-1)
        # 计算 (pred - y)^2，形状保持为 (B,)
        l_sp_vec = (pred - y).pow(2)

        # --- 辅助任务：特征重构 Loss ---
        rec = out.get('rec_feats', None)
        comp = out.get('complete_feats', None)

        if rec is not None and comp is not None:
          # 计算重构误差 MSE
          # mean(dim=(1, 2)) 表示在时间步 (dim 1) 和特征维度 (dim 2) 上取平均
          # 最终得到每个样本一个标量 loss，形状 (B,)
          l_rec_vec = (rec - comp).pow(2).mean(dim=(1, 2))
        else:
          l_rec_vec = torch.zeros_like(l_sp_vec)

        # ==========================
        # 2. 确定最终生效的样本权重
        # ==========================
    
        # 优先使用 sample_weight (它通常已经包含了 sigmoid gate * rank_weight)。
        # 如果没有 weight，尝试使用 mask (仅做 0/1 筛选)。
        # 如果都没有，默认为 1.0 (所有样本权重相等)。
        
        # --- 预测任务权重 ---
        if sample_weight_pred is not None:
            w_eff_pred = sample_weight_pred
        else:
            # 如果没有权重，退化为 Mask 过滤；如果也没 Mask，全为 1
            w_eff_pred = sample_mask_pred if sample_mask_pred is not None else torch.ones_like(l_sp_vec)
         
         # --- 重构任务权重 ---
        if sample_weight_rec is not None:
            w_eff_rec = sample_weight_rec
        else:
            w_eff_rec = sample_mask_rec if sample_mask_rec is not None else torch.ones_like(l_rec_vec)

        
        # ==========================
        # 3. 加权求和 (Weighted Mean)
        # ==========================
        
        # 为什么不能直接 sum() / batch_size？
        # 如果课程学习 Gate 过滤掉了 50% 的难样本（权重为0），那么有效样本数实际上减少了。
        # 如果仍除以 Batch Size，会导致 Loss 数值整体偏小，梯度变弱。
        # 正确做法是：除以“权重的和”，这等价于在“有效样本”上求平均。

        denom_sp = w_eff_pred.sum() + 1e-6  # 加 eps 防止除零
        denom_rec = w_eff_rec.sum() + 1e-6

        # Loss = sum(Loss_i * w_i) / sum(w_i)
        l_sp = (l_sp_vec * w_eff_pred).sum() / denom_sp
        l_rec = (l_rec_vec * w_eff_rec).sum() / denom_rec
        
        # KL Loss 通常作为正则项，一般是对所有样本平均，或者模型内部已经处理好 mean
        # 如果 KL 也需要课程学习，可以在这里加上对应的权重处理逻辑
        l_kl = out.get("kl_loss", torch.tensor(0.0).to(l_sp.device))

        # ==========================
        # 4. 总损失聚合
        # ==========================
        loss = self.sigma * l_sp + self.gamma * l_rec + self.kl * l_kl

        return {'loss': loss,            # 用于反向传播的总 Loss
                'l_sp': l_sp.detach(),   # 记录用：分离计算图的预测 Loss
                'l_rec': l_rec.detach(), # 记录用：分离计算图的重构 Loss
                'l_kl': l_kl.detach()    # 记录用：分离计算图的 KL Loss
                }
