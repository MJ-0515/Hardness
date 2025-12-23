import torch
import torch.nn as nn
import torch.nn.functional as F

"""
多模态课程学习（Curriculum Learning）与难度感知（Hardness-Aware）模块。
  -不仅仅是简单的根据 Loss 排序，而是融合了重构误差、模态不一致性、不确定性等多个维度的“难度”指标，
  -并通过 EMA（指数移动平均）和自适应调度器来动态调整样本权重。
"""
class RunningNorm(nn.Module):
    """
    在线归一化模块 (Running Normalization)。
    
    作用：
        用于解决不同难度指标（如 MSE Loss、Cosine 距离、熵）量纲和数值范围不一致的问题。
        通过维护全局的运行均值 (mean) 和方差 (var)，将输入标准化到 N(0, 1) 分布附近。
        类似于 Batch Normalization，但是用于统计流式数据的累积分布，不参与反向传播更新参数。
    """
    def __init__(self, momentum=0.02, eps=1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        # register_buffer 注册的变量会随模型保存，但不会被优化器更新
        self.register_buffer("mean", torch.zeros(1)) #全局的滑动平均（Running Mean）
        self.register_buffer("var", torch.ones(1)) # 滑动方差（Running Variance）
        self.register_buffer("inited", torch.tensor(0, dtype=torch.long)) # 标记是否已初始化

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """根据当前 Batch 的数据 x，更新全局的 mean 和 var, 通常在训练阶段 (train mode) 的每个 step 调用一次"""
        if x.numel() == 0:
            return
        m = x.mean()
        v = x.var(unbiased=False) + self.eps
        
        # 如果是第一次运行，直接赋值
        if self.inited.item() == 0:
            self.mean.copy_(m)
            self.var.copy_(v)
            self.inited.fill_(1)
        else:
            # 使用动量公式进行 EMA 更新： running = (1-m) * running + m * current,这种方式可以让统计量主要依赖历史积累 (0.98)，同时缓慢吸收新数据 (0.02)
            self.mean.mul_(1 - self.momentum).add_(self.momentum * m)
            self.var.mul_(1 - self.momentum).add_(self.momentum * v)

    def normalize(self, x: torch.Tensor): #执行 Z-Score 标准化 (Standardization), 将数据映射到均值为 0，方差为 1 的分布
        """执行标准化: (x - mean) / std"""
        return (x - self.mean) / torch.sqrt(self.var + self.eps)
        ## 注意：这里使用的是 self.mean (全局历史均值)，而不是 x.mean() (当前 Batch 均值)
        # 这保证了推理的一致性，也避免了 Batch Size 过小带来的抖动。


def _entropy3(w3: torch.Tensor, eps=1e-8):
    """
    计算三分类（或三模态权重）的归一化熵。
    
    Args:
        w3: (B, 3) 权重或概率分布
    Returns:
        ent: (B,) 归一化到 [0, 1] 的熵值。
             值越接近 1 表示分布越均匀（不确定性高/难），越接近 0 表示越尖锐（确定性高/易）。
    """
    w = w3.clamp(min=eps) # 防止 log(0)
    ent = -(w * w.log()).sum(dim=-1)
    # 除以 log(3) 进行归一化，因为 3 分类的最大熵是 log(3)
    ent = ent / torch.log(torch.tensor(3.0, device=w.device))
    return ent.clamp(0.0, 1.0)


class HardnessBank:
    """
    难度记忆库 (Hardness Memory Bank)。
    
    核心思想：
        单个 Batch 内计算出的难度（Loss 或 Error）波动很大且受采样影响。
        该模块利用样本的唯一索引 (Index)，记录每个样本历史难度的 EMA (指数移动平均) 值。
    
    作用：
        1. 跨 Batch 平滑：减少偶然噪声，获得样本真实的“长期”难度。
        2. 拉开区分度：随着训练进行，难样本和简单样本的 EMA 值会逐渐分离。
    """
    def __init__(self, num_samples: int, momentum: float = 0.05, device: str = "cpu"):
        self.num_samples = num_samples # 数据集总样本数
        self.momentum = momentum       # EMA 更新动量
        self.device = device #cpu
        self.h = torch.zeros(num_samples, device=device)   # 存储难度值, [1284,1]
        self.cnt = torch.zeros(num_samples, device=device) # 记录样本被更新的次数

    @torch.no_grad()
    def update(self, indices: torch.Tensor, h_new: torch.Tensor):
        """
        根据当前 Batch 的新计算结果更新记忆库。
        
        Args:
            indices: 当前 Batch 样本在数据集中的绝对索引 (B,)
            h_new: 当前 Batch 计算出的瞬时难度 (B,)
        """
        idx = indices.detach().long().to(self.device)
        val = h_new.detach().float().to(self.device).clamp(0.0, 1.0)
        old = self.h[idx]
        cnt = self.cnt[idx]
        m = self.momentum

        # 逻辑：如果是该样本第一次出现 (cnt < 0.5)，直接赋值，不用 EMA
        # 否则：New = (1-m) * Old + m * Val
        fresh = (cnt < 0.5)
        out = torch.where(fresh, val, (1 - m) * old + m * val)

        self.h[idx] = out
        self.cnt[idx] = cnt + 1

    @torch.no_grad()
    def get(self, indices: torch.Tensor):
        """获取指定索引样本的历史难度值"""
        idx = indices.detach().long().to(self.device)
        return self.h[idx]


class HardnessEstimator(nn.Module):
    """
    P-RMF 专属的多信号难度估计器。
    它不依赖单一指标，而是融合了 5 个维度的信号来评估样本有多“难”：
    
    1. Direct (重构难度): 特征重构误差 (TopK 聚合 + Cosine 距离 + 方差)。
    2. Indirect (模态不一致): 音频、视频、文本的单模态表征之间的距离。如果模态冲突，视为难样本。
    3. Unc (不确定性): 模态融合权重的熵。权重越平均说明模型无法确定哪个模态重要（迷茫）。
    4. Task (任务难度): 主任务的情感预测误差 (Loss)。
    5. Miss (缺失先验): 缺失率越高，天生越难。
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        # 各个难度分量的权重系数
        self.alpha_direct = float(cfg.get("alpha_direct", 0.35)) # Direct 难度权重
        self.alpha_indirect = float(cfg.get("alpha_indirect", 0.20)) # Indirect 难度权重
        self.alpha_unc = float(cfg.get("alpha_unc", 0.15)) # Uncertainty 难度权重
        self.alpha_task = float(cfg.get("alpha_task", 0.20)) # Task 难度权重
        self.alpha_miss = float(cfg.get("alpha_miss", 0.10)) # Missing 先验难度权重

        # Sigmoid 缩放系数，用于将 normalize 后的值映射回 (0, 1) 区间时控制陡峭程度
        self.beta = float(cfg.get("beta", 2.5))
        
        # Direct 计算中的超参， Top-K 策略
        self.topk_ratio = float(cfg.get("topk_ratio", 0.5)) # 只关注重构误差最大的 Top K% Token
        self.use_cos = bool(cfg.get("use_cos", True))       # 是否加入 Cosine 相似度
        self.use_var = bool(cfg.get("use_var", True))       # 是否加入方差惩罚

        # 为不同量纲的指标 分别实例化"在线归一化器"
        self.norm_direct = RunningNorm(momentum=float(cfg.get("rn_momentum", 0.02)))  #
        self.norm_indirect = RunningNorm(momentum=float(cfg.get("rn_momentum", 0.02)))
        self.norm_task = RunningNorm(momentum=float(cfg.get("rn_momentum", 0.02)))

    def _direct(self, out: dict, data: dict):
        """
        计算直接 重构难度。
        逻辑：对比原始完整特征 (complete_feats) 和模型重构特征 (rec_feats)。
        """
        # 异常处理：如果没有重构输出，难度置 0
        if out.get("rec_feats", None) is None or out.get("complete_feats", None) is None:
            bsz = out["sentiment_preds"].shape[0]
            z = torch.zeros(bsz, device=out["sentiment_preds"].device)
            return z

        rec = out["rec_feats"]          # (B, 24, D) -> 总序列长度 24 为三个模态的拼接
        comp = out["complete_feats"]    # (B, 24, D)
        B, T, _ = rec.shape
        tok = T // 3 # 总长 T 切分为三份=8，分别对应 A、V、L 三个模态

        # 三模态拼接长度进行切分为 Audio, Video, Text   (B, 8, 128)
        rec_a, rec_v, rec_l = rec[:, 0:tok], rec[:, tok:2*tok], rec[:, 2*tok:3*tok]
        cmp_a, cmp_v, cmp_l = comp[:, 0:tok], comp[:, tok:2*tok], comp[:, 2*tok:3*tok]

        def agg(x_rec, x_cmp): #计算单模态内的重构难度
            """单模态内部的差异聚合函数"""
            # 1. Token 级 MSE均方误差
            tok_mse = (x_rec - x_cmp).pow(2).mean(dim=-1)  # (B, tok=8), 计算每个Token的误差
            
            # 2. Top-K 聚合：只取误差最大的前 K 个 Token 的均值
            # 解释：某些 Token (如静音段) 容易重构，平均值会掩盖关键帧的重构失败。TopK 更敏感。
            k = max(1, int(self.topk_ratio * tok))  #确定 K 值，例如前 50%=8*0.5=4
            topk = torch.topk(tok_mse, k=k, dim=1, largest=True).values.mean(dim=1)  # (B,)

            score = topk

            # 3. 加上 Cosine 距离 (1 - cos)，关注方向/语义一致性
            if self.use_cos:
                cos = F.cosine_similarity(x_rec, x_cmp, dim=-1)  # (B, tok)
                cos_d = (1.0 - cos).clamp(0.0, 2.0) # 将相似度转换为距离，范围 [0, 2]
                topk_cos = torch.topk(cos_d, k=k, dim=1, largest=True).values.mean(dim=1)
                score = score + 0.5 * topk_cos # 

            # 4. 加上方差项：如果 Token 间的误差波动极大，也视为不稳定/难样本
            if self.use_var:
                tok_var = tok_mse.var(dim=1, unbiased=False)  # (B,)
                score = score + 0.25 * tok_var

            return score
            #tensor([1.4252, 1.4188, 1.4187, 1.4192, 1.4190, 1.4188, 1.4192, 1.4186, 1.4198,1.4200, 1.4190, 1.4190, 1.4188, 1.4187, 1.4189, 1.4191, 1.4189, 1.4186,1.4187, 1.4185, 1.4185, 1.4190, 1.4190, 1.4215], device='cuda:0')
        # 三个模态各自的难度分数
        s_a = agg(rec_a, cmp_a)  #(B,)
        s_v = agg(rec_v, cmp_v)
        s_l = agg(rec_l, cmp_l)

        # 基于缺失率 (Missing Rate) 的动态加权: 如果某模态缺失率高，该模态的重构误差在总分中占比应更大 (或根据具体逻辑调整)
        # 这里逻辑是：缺失率高(w大) -> 更加重该模态的重构考核权重
        mr_a = data["labels"]["missing_rate_a"].to(s_a.device).view(-1)  #(B,)
        mr_v = data["labels"]["missing_rate_v"].to(s_a.device).view(-1)
        mr_l = data["labels"]["missing_rate_l"].to(s_a.device).view(-1)

        w_a = 1.0 + mr_a
        w_v = 1.0 + mr_v
        w_l = 1.0 + mr_l

        direct = (w_a * s_a + w_v * s_v + w_l * s_l) / (w_a + w_v + w_l + 1e-6) #(B,)
        return direct

    def _indirect(self, out: dict):
        """
        计算间接难度（模态不一致性）。
        逻辑：如果同一个样本的 Audio, Video, Text 特征向量在空间中距离很远，说明模态间语义冲突或噪声大。
        """
        if out.get("h_1_a", None) is None:
            bsz = out["sentiment_preds"].shape[0]
            return torch.zeros(bsz, device=out["sentiment_preds"].device)

        # 获取单模态编码器的最终表征 (Mean Pooling), mean(dim=1) 是在时间维度上求平均
        ha = out["h_1_a"].mean(dim=1) # [B, 128]
        hv = out["h_1_v"].mean(dim=1)
        hl = out["h_1_l"].mean(dim=1)

        def cos_d(x, y): # F.cosine_similarity 输出范围 [-1, 1], 1.0 - similarity 输出范围 [0, 2]
            return (1.0 - F.cosine_similarity(x, y, dim=-1)).clamp(0.0, 2.0)

        # 计算两两之间的距离
        d_av = cos_d(ha, hv) #(B) 
        d_al = cos_d(ha, hl)
        d_vl = cos_d(hv, hl)

        return (d_av + d_al + d_vl) / 3.0  #取平均值作为最终的间接难度分数

    def _uncertainty(self, out: dict):
        """
        计算不确定性。
        逻辑：基于融合模块分配给 T/V/A 的权重熵。
        """
        w = out.get("weight_t_v_a", None)
        if w is None:
            bsz = out["sentiment_preds"].shape[0]
            return torch.zeros(bsz, device=out["sentiment_preds"].device)
        # 将权重压缩为 (B, 3) -> [text_w, visual_w, audio_w]
        w3 = w.mean(dim=(2, 3)).transpose(0, 1).contiguous()  #在 Time(2) 和 Feature(3) 维度上取平均
        return _entropy3(w3)

    def _task(self, out: dict, label: dict):
        """任务难度：简单的 MSE Residual"""
        pred = out["sentiment_preds"].view(-1)
        y = label["sentiment_labels"].view(-1)
        return (pred - y).pow(2)

    def _miss(self, data: dict, device):
        """先验难度：直接取三模态缺失率的平均值"""
        mr_a = data["labels"]["missing_rate_a"].to(device).view(-1)
        mr_v = data["labels"]["missing_rate_v"].to(device).view(-1)
        mr_l = data["labels"]["missing_rate_l"].to(device).view(-1)
        return (mr_a + mr_v + mr_l) / 3.0

    def forward(self, out: dict, label: dict, data: dict, is_train: bool = True):
        """
        前向计算综合难度。
        Returns:
            h: (B,) 综合难度值 [0, 1]
            details: 字典，包含各分量原始值，用于日志记录
        """
        device = out["sentiment_preds"].device

        # 1. 计算各项原始指标
        direct = self._direct(out, data)  #重构难度
        indirect = self._indirect(out)  #模态不一致难度
        unc = self._uncertainty(out)  #不确定性难度
        task = self._task(out, label)  #任务难度
        miss = self._miss(data, device)  #缺失先验难度0

        # 2. 更新归一化统计量 (仅在训练时)
        if is_train:
            with torch.no_grad():
                self.norm_direct.update(direct.detach())
                self.norm_indirect.update(indirect.detach())
                self.norm_task.update(task.detach())

        # 3. 归一化并 Sigmoid 映射到 (0, 1)
        # beta 控制 sigmoid 的斜率，beta 越大，区分度越明显
        direct_n = torch.sigmoid(self.beta * self.norm_direct.normalize(direct))
        indirect_n = torch.sigmoid(self.beta * self.norm_indirect.normalize(indirect))
        task_n = torch.sigmoid(self.beta * self.norm_task.normalize(task))

        # 4. 加权求和得到最终综合难度
        h = (
            self.alpha_direct * direct_n +
            self.alpha_indirect * indirect_n +
            self.alpha_unc * unc +
            self.alpha_task * task_n +
            self.alpha_miss * miss
        )
#tensor([0.5859, 0.2690, 0.5676, 0.5274, 0.4365, 0.5802, 0.6664, 0.5303, 0.5692,
       # 0.7947, 0.2883, 0.5877, 0.6119, 0.4795, 0.4162, 0.3925, 0.5945, 0.4936,
        #0.5895, 0.7167, 0.6361, 0.3898, 0.5041, 0.5811], device='cuda:0')
        return h.clamp(0.0, 1.0), {
            "direct": direct.detach(),
            "indirect": indirect.detach(),
            "unc": unc.detach(),
            "task": task.detach(),
            "miss": miss.detach()
        }


def _rank01(x: torch.Tensor):
    """
    计算输入张量中每个元素在 Batch 内的相对排名 (百分位)。
    
    Args:
        x: (B,) 输入数值
    Returns:
        ranks: (B,) 范围 [0, 1]。最小值为 0.0，最大值为 1.0。
    
    作用：
        相比直接使用 Loss 值，使用 Rank (排名) 对异常值不敏感，且分布更均匀，
        使得加权逻辑更稳定（Robustness）。
    """
    B = x.numel()
    if B <= 1:
        return torch.zeros_like(x)
    order = torch.argsort(x)                 # 获取从小到大的索引
    ranks = torch.empty_like(order, dtype=torch.float)
    # 将排名赋回原位置
    ranks[order] = torch.arange(B, device=x.device, dtype=torch.float)
    return ranks / float(B - 1)

class AdaptiveHardnessScheduler:
    """
    自适应课程学习调度器 (Curriculum Learning Scheduler)。
    
    核心逻辑：
    1. Pacing (步调): 随着 epoch 增加，逐渐增加训练样本的难度阈值 (q 从 0.3 -> 1.0)。
    2. Gating (门控): 利用 Soft Gating 机制，给极其困难（超过当前能力范围）的样本极小的权重。
    3. Reweighting (重加权): 在通过门控的样本中，根据 Rank 进行多项式加权，让模型更关注“当前阶段较难”的样本。
    """
    def __init__(self, cfg: dict, total_epochs: int):
        self.total_epochs = total_epochs
        
        # 热身阶段：前几个 epoch 不进行加权干预
        self.warmup_epochs = int(cfg.get("warmup_epochs", 3))

        # 课程进度控制：有效样本比例 q 从 q_start 增加到 q_end
        self.q_start = float(cfg.get("q_start", 0.30))
        self.q_end = float(cfg.get("q_end", 1.00))

        # 温度系数：控制 Gate 的软硬程度。
        # start时温度高(0.15) -> 边界模糊；end时温度低(0.05) -> 边界清晰(近似阶跃)
        self.temp_start = float(cfg.get("temp_start", 0.15))
        self.temp_end = float(cfg.get("temp_end", 0.05))

        # 重加权参数
        self.eta_max = float(cfg.get("eta_max", 0.6)) # 最大加权强度
        self.p_pred = float(cfg.get("p_pred", 2.0))   # 预测任务的聚焦指数 (Focusing Parameter)
        self.p_rec = float(cfg.get("p_rec", 1.5))     # 重构任务的聚焦指数

        # 权重裁剪范围，防止梯度爆炸或消失
        self.w_clip_min = float(cfg.get("w_clip_min", 0.2))
        self.w_clip_max = float(cfg.get("w_clip_max", 5.0))

        # 是否对重构 Loss 也应用门控机制 (通常重构任务不需要像分类任务那样严格的课程)
        self.gate_rec = bool(cfg.get("gate_rec", False))

        # 关键开关：是否分离难度源。
        # True: 预测任务看 Task 难度，重构任务看 Direct 难度。
        # False: 都看综合难度 h_all。
        self.use_split_sources = bool(cfg.get("use_split_sources", True))

    def _progress(self, epoch:int):
        """计算当前训练进度 [0.0, 1.0]，扣除 Warmup"""
        if epoch <= self.warmup_epochs:
            return 0.0
        denom = max(1, self.total_epochs - self.warmup_epochs)
        # 计算当前进度
        prog = float(epoch - self.warmup_epochs) / float(denom)
        return prog

    def _anneal(self, a: float, b: float, t: float):
        """线性退火辅助函数"""
        return a + (b - a) * t

    @torch.no_grad()
    def map(self, h_all: torch.Tensor, epoch: int,
            h_direct: torch.Tensor = None, h_task: torch.Tensor = None):
        """
        核心映射函数：将难度值映射为 Loss 权重, 决定了模型在当前 Epoch 应该学习哪些样本，以及应该多关注哪些样本  
        Args:
            h_all: (B,) 综合难度 [0,1]，来自 HardnessBank; epoch: 当前 epoch; h_direct: (B,) Direct 难度 (可选);h_task: (B,) Task 难度 (可选)
        Returns:
            w_pred: (B,) 主任务 Loss 的权重; w_rec: (B,) 重构任务 Loss 的权重
            g_pred: (B,) 预测任务的门控值 (用于统计被丢弃的样本比例);g_rec: (B,) 重构任务的门控值
        """

        device = h_all.device
        prog = self._progress(epoch)  # 计算进度 0.0 -> 1.0

        # Warmup 期间，权重全部为 1.0，不做干预
        if prog <= 0.0:
            ones = torch.ones_like(h_all, device=device)
            return ones, ones, ones, ones
        

        # 计算当前步的超参 (线性退火)
        q = self._anneal(self.q_start, self.q_end, prog)      # 当前允许通过的难度分位数 (0.3 -> 1.0)
        temp = self._anneal(self.temp_start, self.temp_end, prog) # 温度系数
        eta = self.eta_max * prog                             # 加权强度随训练增加

        # 1. 确定驱动源 (Source Selection)
        if self.use_split_sources and (h_direct is not None) and (h_task is not None):
            src_pred = h_task    # 预测任务权重由 Task 难度决定
            src_rec = h_direct   # 重构任务权重由 Direct 难度决定
        else:
            src_pred = h_all
            src_rec = h_all

        # 2. 课程门控 (Curriculum Gating)
        # 这里的逻辑是：找出当前 Batch 中处于 q 分位数的难度值 tau
        # 难度 > tau 的样本，会被 sigmoid 压低权重 (被视为“太难而无法学习”的样本)
        tau_pred = torch.quantile(src_pred, q=q).item()
        # Soft Gate 公式: sigmoid((阈值 - 难度) / 温度)
        # 如果 难度 < 阈值，值 > 0，sigmoid -> 1 (通过)
        # 如果 难度 > 阈值，值 < 0，sigmoid -> 0 (抑制)
        g_pred = torch.sigmoid((tau_pred - src_pred) / max(temp, 1e-6))

        if self.gate_rec:
            tau_rec = torch.quantile(src_rec, q=q).item()
            g_rec = torch.sigmoid((tau_rec - src_rec) / max(temp, 1e-6))
        else:
            g_rec = torch.ones_like(g_pred)

        # 3. Rank-based Reweighting (基于排名的重加权)
        # 对通过门控的样本，我们希望重点关注比较难的那部分（即处于 Curriculum 边缘的样本）
        # 公式: w = (1-eta) + eta * (rank^p)
        # p > 1 时，rank 越高（越难），权重越大
        r_pred = _rank01(src_pred)
        r_rec = _rank01(src_rec)

        w_pred = (1.0 - eta) + eta * (r_pred.clamp(1e-6, 1.0) ** self.p_pred)
        w_rec = (1.0 - eta) + eta * (r_rec.clamp(1e-6, 1.0) ** self.p_rec)

        # 4. 组合：最终权重 = 重加权系数 * 门控系数
        w_pred = w_pred * g_pred
        w_rec = w_rec * g_rec

        # 5. 保持均值为 1 (Mean-Preserving Normalization)
        # 避免改变 Loss 的整体量级，只改变样本间的相对贡献
        

        # 6. 安全裁剪
        w_pred = w_pred.clamp(self.w_clip_min, self.w_clip_max)
        w_rec = w_rec.clamp(self.w_clip_min, self.w_clip_max)

        w_pred = w_pred / (w_pred.mean() + 1e-6)
        w_rec = w_rec / (w_rec.mean() + 1e-6)

        return w_pred, w_rec, g_pred, g_rec
