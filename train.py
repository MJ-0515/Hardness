import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results, interval_time, get_parameter_number
from models.P_RMF import build_model
from core.metric import MetricsTop
from tqdm import tqdm
import time
import csv
from pathlib import Path

#from torch.utils.data import DataLoader, WeightedRandomSampler
from models.hardness import HardnessEstimator, HardnessBank, AdaptiveHardnessScheduler

from core.hard_log import hardness_diagnostics

start = time.time()
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)
# 运行前设置的seed和yaml路径传给opt
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='')
parser.add_argument('--seed', type=int, default=-1)
opt = parser.parse_args()
print("-------------------------------------------------------------------------------")
print(opt)    #Namespace(config_file='configs/train_mosi.yaml', seed=1111)
print("-------------------------------------------------------------------------------")

def _quantiles_1d(x: torch.Tensor, qs=(0.10, 0.50, 0.90)):
    """
    x: 1D tensor on CPU
    return: list of floats
    """
    if x.numel() == 0:
        return [float("nan")] * len(qs)
    q = torch.tensor(list(qs), dtype=torch.float32)
    return [float(v) for v in torch.quantile(x.float(), q).tolist()]


def main():
    best_valid_results, best_test_results = {}, {}

    config_file = 'configs/train_mosi.yaml' if opt.config_file == '' else opt.config_file
    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    os.makedirs(ckpt_root, exist_ok=True)

    stats_csv = os.path.join(ckpt_root, f"hardness_stats_seed{seed}.csv")


    model = build_model(args).to(device)
    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.AdamW([
        {'params': model.bertmodel.parameters(), 'lr': 0.000005},
        {'params': model.crossmodal_encoder.parameters(), 'lr': 0.000005},
        {'params': model.proj_l.parameters(), 'lr': args['base']['lr']},
        {'params': model.proj_a.parameters(), 'lr': args['base']['lr']},
        {'params': model.proj_v.parameters(), 'lr': args['base']['lr']},
        {'params': model.generate_proxy_modality.parameters(), 'lr': args['base']['lr']},
        {'params': model.fc1.parameters(), 'lr': args['base']['lr']},
        {'params': model.fc2.parameters(), 'lr': args['base']['lr']},
    ], weight_decay=args['base']['weight_decay'])

    scheduler_warmup = get_scheduler(optimizer, args)
    loss_fn = MultimodalLoss(args)
    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    # 1) 初始化难度模块
    use_hard = bool(args.get("hardness", {}).get("enable", True))
    hard_est, hard_bank, hard_sched = None, None, None
    if use_hard:
        hard_est = HardnessEstimator(args["hardness"]["estimator"]).to(device)

        # bank 放 CPU 即可，节省显存
        hard_bank = HardnessBank(
            num_samples=len(dataLoader["train"].dataset),
            momentum=float(args["hardness"]["bank"].get("momentum", 0.05)),
            device="cpu"
        )

        hard_sched = AdaptiveHardnessScheduler(
            args["hardness"]["scheduler"],
            total_epochs=int(args["base"]["n_epochs"])
        )
    print("----------------------------------------------开始训练！！！---------------------------------------------------------")
    for epoch in range(1, args['base']['n_epochs'] + 1):
        print(f'Training Epoch: {epoch}')
        start_time = time.time()

        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))
        #------------------------------！！！进入 train   ！！！--------------------------------
        hard_stats = train(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
            metrics=metrics,
            use_hard=use_hard,
            hard_est=hard_est,
            hard_bank=hard_bank,
            hard_sched=hard_sched,
            stats_csv_path=stats_csv
        )

        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(
                valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed,
                'valid', save_best_model=False
            )
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(
            test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed,
            'test', save_best_model=True
        )

        epoch_mins, epoch_secs = interval_time(start_time, time.time())
        print(
            "Epoch: {}/{} | Current Best Test Results: {} |\nTime: {}m {}s".format(
                epoch, args['base']['n_epochs'], best_test_results, epoch_mins, epoch_secs
            )
        )

        scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn, epoch, metrics,
          use_hard=False, hard_est=None, hard_bank=None, hard_sched=None,
          stats_csv_path=None):
    y_pred, y_true = [], []
    loss_dict = {}

    # 统计容器
    eff_pred_sum = 0.0
    eff_pred_cnt = 0
    w_pred_buf = []
    h_all_buf = []

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}

        out = model(complete_input, incomplete_input)

        # 默认权重与门控
        w_pred = None
        w_rec = None
        m_pred = None
        m_rec = None
        h_all = None  # 为统计输出保留

        if use_hard:
            indices = data['index']
            if not torch.is_tensor(indices):
                indices = torch.tensor(indices, dtype=torch.long)

            with torch.no_grad():
                # 1) 计算当前 batch 难度
                h_now, parts = hard_est(out, label, data, is_train=True)

                # 2) 更新并取 bank 中平滑后的综合难度
                hard_bank.update(indices, h_now)
                h_all = hard_bank.get(indices).to(device)

                # 3) 为 split-source 映射准备 h_direct 与 h_task
                beta = float(getattr(hard_est, "beta", 2.5))
                direct_raw = parts["direct"].to(device)
                task_raw = parts["task"].to(device)

                h_direct = torch.sigmoid(beta * hard_est.norm_direct.normalize(direct_raw))
                h_task = torch.sigmoid(beta * hard_est.norm_task.normalize(task_raw))

                # 4) 映射得到 w_pred/w_rec 以及门控 m_pred/m_rec
                w_pred, w_rec, m_pred, m_rec = hard_sched.map(
                    h_all=h_all,
                    epoch=epoch,
                    h_direct=h_direct,
                    h_task=h_task
                )

                w_pred = w_pred.detach()
                w_rec = w_rec.detach()
                m_pred = m_pred.detach()
                m_rec = m_rec.detach()

                # 5) 收集统计数据（每个 batch）
                eff_pred_sum += float(m_pred.mean().item())
                eff_pred_cnt += 1
                w_pred_buf.append(w_pred.detach().cpu().view(-1))
                h_all_buf.append(h_all.detach().cpu().view(-1))

        # 计算 loss（分损失权重版本）
        loss = loss_fn(
            out,
            label,
            sample_weight_pred=w_pred,
            sample_weight_rec=w_rec,
            sample_mask_pred=m_pred,
            sample_mask_rec=m_rec
        )

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].detach().cpu())
        y_true.append(label['sentiment_labels'].detach().cpu())

        # loss 统计
        if cur_iter == 0:
            for k, v in loss.items():
                loss_dict[k] = float(v.item()) if torch.is_tensor(v) else float(v)
        else:
            for k, v in loss.items():
                loss_dict[k] += float(v.item()) if torch.is_tensor(v) else float(v)

    # epoch 汇总指标（原有）
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    loss_dict = {k: v / (cur_iter + 1) for k, v in loss_dict.items()}
    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')

    # epoch 汇总统计（新增）
    hardness_epoch_stats = None
    if use_hard and eff_pred_cnt > 0 and len(w_pred_buf) > 0 and len(h_all_buf) > 0:
        w_pred_all = torch.cat(w_pred_buf, dim=0)
        h_all_all = torch.cat(h_all_buf, dim=0)

        eff_pred_mean = eff_pred_sum / eff_pred_cnt
        wq10, wq50, wq90 = _quantiles_1d(w_pred_all, qs=(0.10, 0.50, 0.90))
        hq10, hq50, hq90 = _quantiles_1d(h_all_all, qs=(0.10, 0.50, 0.90))

        print(
            f"[Hardness][Epoch {epoch}] "
            f"eff_pred_mean={eff_pred_mean:.4f} | "
            f"w_pred_p10/p50/p90={wq10:.4f}/{wq50:.4f}/{wq90:.4f} | "
            f"h_all_p10/p50/p90={hq10:.4f}/{hq50:.4f}/{hq90:.4f}"
        )

        if stats_csv_path is not None:
            stats_csv_path = Path(stats_csv_path)
            stats_csv_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = stats_csv_path.exists()

            with open(stats_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "epoch",
                        "eff_pred_mean",
                        "w_pred_p10", "w_pred_p50", "w_pred_p90",
                        "h_all_p10", "h_all_p50", "h_all_p90"
                    ])
                writer.writerow([
                    epoch,
                    f"{eff_pred_mean:.6f}",
                    f"{wq10:.6f}", f"{wq50:.6f}", f"{wq90:.6f}",
                    f"{hq10:.6f}", f"{hq50:.6f}", f"{hq90:.6f}"
                ])

        hardness_epoch_stats = {
            "eff_pred_mean": eff_pred_mean,
            "w_pred_p10": wq10, "w_pred_p50": wq50, "w_pred_p90": wq90,
            "h_all_p10": hq10, "h_all_p50": hq50, "h_all_p90": hq90
        }

    return hardness_epoch_stats



def evaluate(model, eval_loader, loss_fn, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()

    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}
        with torch.no_grad():
            out = model(complete_input, incomplete_input)

        loss = loss_fn(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item()
                except:
                    loss_dict[key] += value

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    return results


if __name__ == '__main__':
    main()
print("Time Usage: {} minutes {} seconds".format(*interval_time(start, time.time())))
