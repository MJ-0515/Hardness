# core/hard_log.py

#简单的 hardness 可视化日志
import torch


def hardness_diagnostics(global_hardness, epoch, args=None):
    """
    打印当前 epoch 的 hardness 分布统计，方便你观察 curriculum 是否生效。

    global_hardness: [N] 的 Tensor 或 list，每个训练样本一个难度值
    """
    if isinstance(global_hardness, torch.Tensor):
        h = global_hardness.detach().cpu()
    else:
        h = torch.tensor(global_hardness)

    mean = h.mean().item()
    min_v = h.min().item()
    max_v = h.max().item()
    q25 = h.quantile(0.25).item()
    q50 = h.quantile(0.5).item()
    q75 = h.quantile(0.75).item()

    print(
        f"[Epoch {epoch}] Hardness stats -> "
        f"mean={mean:.4f}, min={min_v:.4f}, max={max_v:.4f}, "
        f"q25={q25:.4f}, median={q50:.4f}, q75={q75:.4f}"
    )
