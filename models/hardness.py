import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNorm(nn.Module):
    def __init__(self, momentum=0.02, eps=1e-6):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("var", torch.ones(1))
        self.register_buffer("inited", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        m = x.mean()
        v = x.var(unbiased=False) + self.eps
        if self.inited.item() == 0:
            self.mean.copy_(m)
            self.var.copy_(v)
            self.inited.fill_(1)
        else:
            self.mean.mul_(1 - self.momentum).add_(self.momentum * m)
            self.var.mul_(1 - self.momentum).add_(self.momentum * v)

    def normalize(self, x: torch.Tensor):
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


def _entropy3(w3: torch.Tensor, eps=1e-8):
    # w3: (B, 3)
    w = w3.clamp(min=eps)
    ent = -(w * w.log()).sum(dim=-1)
    ent = ent / torch.log(torch.tensor(3.0, device=w.device))
    return ent.clamp(0.0, 1.0)


class HardnessBank:
    """
    基于样本 index 的 EMA 难度记忆库。
    作用：跨 batch 平滑 + 拉开区分度。
    """
    def __init__(self, num_samples: int, momentum: float = 0.05, device: str = "cpu"):
        self.num_samples = num_samples
        self.momentum = momentum
        self.device = device
        self.h = torch.zeros(num_samples, device=device)
        self.cnt = torch.zeros(num_samples, device=device)

    @torch.no_grad()
    def update(self, indices: torch.Tensor, h_new: torch.Tensor):
        idx = indices.detach().long().to(self.device)
        val = h_new.detach().float().to(self.device).clamp(0.0, 1.0)
        old = self.h[idx]
        cnt = self.cnt[idx]
        m = self.momentum

        # 初次出现的样本直接赋值，后续 EMA
        fresh = (cnt < 0.5)
        out = torch.where(fresh, val, (1 - m) * old + m * val)

        self.h[idx] = out
        self.cnt[idx] = cnt + 1

    @torch.no_grad()
    def get(self, indices: torch.Tensor):
        idx = indices.detach().long().to(self.device)
        return self.h[idx]


class HardnessEstimator(nn.Module):
    """
    适配 P-RMF 的多信号难度估计：
    direct: 重构误差 topk 聚合 + cosine + 方差
    indirect: 模态不一致性
    unc: PMG 权重熵
    task: 主任务残差
    miss: 缺失率先验
    """
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        self.alpha_direct = float(cfg.get("alpha_direct", 0.35))
        self.alpha_indirect = float(cfg.get("alpha_indirect", 0.20))
        self.alpha_unc = float(cfg.get("alpha_unc", 0.15))
        self.alpha_task = float(cfg.get("alpha_task", 0.20))
        self.alpha_miss = float(cfg.get("alpha_miss", 0.10))

        self.beta = float(cfg.get("beta", 2.5))
        self.topk_ratio = float(cfg.get("topk_ratio", 0.5))
        self.use_cos = bool(cfg.get("use_cos", True))
        self.use_var = bool(cfg.get("use_var", True))

        self.norm_direct = RunningNorm(momentum=float(cfg.get("rn_momentum", 0.02)))
        self.norm_indirect = RunningNorm(momentum=float(cfg.get("rn_momentum", 0.02)))
        self.norm_task = RunningNorm(momentum=float(cfg.get("rn_momentum", 0.02)))

    def _direct(self, out: dict, data: dict):
        if out.get("rec_feats", None) is None or out.get("complete_feats", None) is None:
            bsz = out["sentiment_preds"].shape[0]
            z = torch.zeros(bsz, device=out["sentiment_preds"].device)
            return z

        rec = out["rec_feats"]          # (B, 24, D)
        comp = out["complete_feats"]    # (B, 24, D)
        B, T, _ = rec.shape
        tok = T // 3

        rec_a, rec_v, rec_l = rec[:, 0:tok], rec[:, tok:2*tok], rec[:, 2*tok:3*tok]
        cmp_a, cmp_v, cmp_l = comp[:, 0:tok], comp[:, tok:2*tok], comp[:, 2*tok:3*tok]

        def agg(x_rec, x_cmp):
            # token 级 MSE
            tok_mse = (x_rec - x_cmp).pow(2).mean(dim=-1)  # (B, tok)
            k = max(1, int(self.topk_ratio * tok))
            topk = torch.topk(tok_mse, k=k, dim=1, largest=True).values.mean(dim=1)  # (B,)

            score = topk

            if self.use_cos:
                cos = F.cosine_similarity(x_rec, x_cmp, dim=-1)  # (B, tok)
                cos_d = (1.0 - cos).clamp(0.0, 2.0)
                topk_cos = torch.topk(cos_d, k=k, dim=1, largest=True).values.mean(dim=1)
                score = score + 0.5 * topk_cos

            if self.use_var:
                score = score + 0.25 * tok_mse.var(dim=1, unbiased=False)

            return score

        s_a = agg(rec_a, cmp_a)
        s_v = agg(rec_v, cmp_v)
        s_l = agg(rec_l, cmp_l)

        # 缺失率越高，该模态重构误差贡献越大
        mr_a = data["labels"]["missing_rate_a"].to(s_a.device).view(-1)
        mr_v = data["labels"]["missing_rate_v"].to(s_a.device).view(-1)
        mr_l = data["labels"]["missing_rate_l"].to(s_a.device).view(-1)

        w_a = 1.0 + mr_a
        w_v = 1.0 + mr_v
        w_l = 1.0 + mr_l

        direct = (w_a * s_a + w_v * s_v + w_l * s_l) / (w_a + w_v + w_l + 1e-6)
        return direct

    def _indirect(self, out: dict):
        # 需要模型 forward 返回 h_1_a/h_1_v/h_1_l
        if out.get("h_1_a", None) is None:
            bsz = out["sentiment_preds"].shape[0]
            return torch.zeros(bsz, device=out["sentiment_preds"].device)

        ha = out["h_1_a"].mean(dim=1)
        hv = out["h_1_v"].mean(dim=1)
        hl = out["h_1_l"].mean(dim=1)

        def cos_d(x, y):
            return (1.0 - F.cosine_similarity(x, y, dim=-1)).clamp(0.0, 2.0)

        d_av = cos_d(ha, hv)
        d_al = cos_d(ha, hl)
        d_vl = cos_d(hv, hl)

        return (d_av + d_al + d_vl) / 3.0

    def _uncertainty(self, out: dict):
        # 需要模型 forward 返回 weight_t_v_a: (3, B, tok, dim_latent)
        w = out.get("weight_t_v_a", None)
        if w is None:
            bsz = out["sentiment_preds"].shape[0]
            return torch.zeros(bsz, device=out["sentiment_preds"].device)
        # 压缩成 (B, 3)
        w3 = w.mean(dim=(2, 3)).transpose(0, 1).contiguous()
        return _entropy3(w3)

    def _task(self, out: dict, label: dict):
        pred = out["sentiment_preds"].view(-1)
        y = label["sentiment_labels"].view(-1)
        return (pred - y).pow(2)

    def _miss(self, data: dict, device):
        mr_a = data["labels"]["missing_rate_a"].to(device).view(-1)
        mr_v = data["labels"]["missing_rate_v"].to(device).view(-1)
        mr_l = data["labels"]["missing_rate_l"].to(device).view(-1)
        return (mr_a + mr_v + mr_l) / 3.0

    def forward(self, out: dict, label: dict, data: dict, is_train: bool = True):
        device = out["sentiment_preds"].device

        direct = self._direct(out, data)
        indirect = self._indirect(out)
        unc = self._uncertainty(out)
        task = self._task(out, label)
        miss = self._miss(data, device)

        if is_train:
            with torch.no_grad():
                self.norm_direct.update(direct.detach())
                self.norm_indirect.update(indirect.detach())
                self.norm_task.update(task.detach())

        direct_n = torch.sigmoid(self.beta * self.norm_direct.normalize(direct))
        indirect_n = torch.sigmoid(self.beta * self.norm_indirect.normalize(indirect))
        task_n = torch.sigmoid(self.beta * self.norm_task.normalize(task))

        h = (
            self.alpha_direct * direct_n +
            self.alpha_indirect * indirect_n +
            self.alpha_unc * unc +
            self.alpha_task * task_n +
            self.alpha_miss * miss
        )

        return h.clamp(0.0, 1.0), {
            "direct": direct.detach(),
            "indirect": indirect.detach(),
            "unc": unc.detach(),
            "task": task.detach(),
            "miss": miss.detach()
        }


def _rank01(x: torch.Tensor):
    # 返回每个元素的 rank 归一化到 [0,1]，对并列值用稳定排序近似处理
    # x: (B,)
    B = x.numel()
    if B <= 1:
        return torch.zeros_like(x)
    order = torch.argsort(x)                 # 从小到大
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(B, device=x.device, dtype=torch.float)
    return ranks / float(B - 1)

class AdaptiveHardnessScheduler:
    """
    改进版映射逻辑：
    1) tau 用分位数自适应，控制有效样本比例
    2) 重加权基于 rank，弱化对绝对值敏感
    3) 输出 w_pred 与 w_rec，用于分损失加权
    """
    def __init__(self, cfg: dict, total_epochs: int):
        self.total_epochs = total_epochs

        self.warmup_epochs = int(cfg.get("warmup_epochs", 3))

        # 课程比例 q 从 q_start -> 1.0
        self.q_start = float(cfg.get("q_start", 0.30))
        self.q_end = float(cfg.get("q_end", 1.00))

        # soft gate 温度退火，前期更平滑，后期更接近硬门控
        self.temp_start = float(cfg.get("temp_start", 0.15))
        self.temp_end = float(cfg.get("temp_end", 0.05))

        # 重加权强度 eta 与形状参数 p
        self.eta_max = float(cfg.get("eta_max", 0.6))
        self.p_pred = float(cfg.get("p_pred", 2.0))
        self.p_rec = float(cfg.get("p_rec", 1.5))

        # 权重裁剪与归一化
        self.w_clip_min = float(cfg.get("w_clip_min", 0.2))
        self.w_clip_max = float(cfg.get("w_clip_max", 5.0))

        # 是否对重构也做课程门控
        self.gate_rec = bool(cfg.get("gate_rec", False))

        # 允许你用不同难度源分别驱动 pred 与 rec
        # 例如 pred 用综合难度 h_all，rec 用 direct 难度 h_direct
        self.use_split_sources = bool(cfg.get("use_split_sources", True))

    def _progress(self, epoch: int):
        if epoch <= self.warmup_epochs:
            return 0.0
        denom = max(1, self.total_epochs - self.warmup_epochs)
        return float(epoch - self.warmup_epochs) / float(denom)

    def _anneal(self, a: float, b: float, t: float):
        return a + (b - a) * t

    @torch.no_grad()
    def map(self, h_all: torch.Tensor, epoch: int,
            h_direct: torch.Tensor = None, h_task: torch.Tensor = None):
        """
        输入
        h_all: (B,) 综合难度 [0,1]，通常来自 HardnessBank
        h_direct: (B,) direct 难度，可选
        h_task: (B,) task 难度，可选

        输出
        w_pred: (B,) 用于主任务损失
        w_rec: (B,) 用于重构损失
        g_pred: (B,) 课程门控
        g_rec: (B,) 重构门控
        """
        device = h_all.device
        prog = self._progress(epoch)

        if prog <= 0.0:
            ones = torch.ones_like(h_all, device=device)
            return ones, ones, ones, ones

        q = self._anneal(self.q_start, self.q_end, prog)
        temp = self._anneal(self.temp_start, self.temp_end, prog)
        eta = self.eta_max * prog

        # 选择驱动源
        if self.use_split_sources and (h_direct is not None) and (h_task is not None):
            src_pred = h_task
            src_rec = h_direct
        else:
            src_pred = h_all
            src_rec = h_all

        # 自适应分位数阈值
        tau_pred = torch.quantile(src_pred, q=q).item()
        g_pred = torch.sigmoid((tau_pred - src_pred) / max(temp, 1e-6))

        if self.gate_rec:
            tau_rec = torch.quantile(src_rec, q=q).item()
            g_rec = torch.sigmoid((tau_rec - src_rec) / max(temp, 1e-6))
        else:
            g_rec = torch.ones_like(g_pred)

        # rank 重加权
        r_pred = _rank01(src_pred)
        r_rec = _rank01(src_rec)

        w_pred = (1.0 - eta) + eta * (r_pred.clamp(1e-6, 1.0) ** self.p_pred)
        w_rec = (1.0 - eta) + eta * (r_rec.clamp(1e-6, 1.0) ** self.p_rec)

        # 组合门控
        w_pred = w_pred * g_pred
        w_rec = w_rec * g_rec

        # 归一化保持整体尺度
        w_pred = w_pred / (w_pred.mean() + 1e-6)
        w_rec = w_rec / (w_rec.mean() + 1e-6)

        # clip 防止极端权重
        w_pred = w_pred.clamp(self.w_clip_min, self.w_clip_max)
        w_rec = w_rec.clamp(self.w_clip_min, self.w_clip_max)

        return w_pred, w_rec, g_pred, g_rec
