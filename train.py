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

#from torch.utils.data import DataLoader, WeightedRandomSampler
from models.hardness import HardnessEstimator #, HardnessScheduler
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

def map_hardness_to_weights(h, mode='linear_center', alpha=0.5, device='cuda'):
    """
    h: [B] ∈ [0,1]
    mode: 'linear_center' / 'power' / 'piecewise'
    """
    h = h.to(device)

    if mode == 'linear_center':
        # 方案 A
        h_centered = h - h.mean()
        w = 1.0 + alpha * h_centered
        max_delta = 0.5
        w = torch.clamp(w, 1.0 - max_delta, 1.0 + max_delta)

    elif mode == 'power':
        # 方案 B
        beta = alpha  # 这里用 alpha 当 beta
        eps = 1e-6
        w = torch.pow(h + eps, beta)
        if torch.all(w <= 0):
            w = torch.ones_like(w)

    elif mode == 'piecewise':
        # 方案 C
        t_low = 0.3
        t_high = 0.8
        w = torch.ones_like(h)
        mask_easy = h < t_low
        mask_mid = (h >= t_low) & (h <= t_high)
        mask_too_hard = h > t_high

        w[mask_easy] = 0.8
        w[mask_mid] = 1.0 + alpha * (h[mask_mid] - t_low) / (t_high - t_low + 1e-8)
        w[mask_too_hard] = 1.0

    else:
        w = torch.ones_like(h)

    # 归一化到 mean=1
    w = w / (w.mean() + 1e-8)
    return w


def main():
    best_valid_results, best_test_results = {}, {}

    config_file = 'configs/train_mosi.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print("************************train.yaml中的内容**************************")
    print(args) 
    #最终seed还是看train.yaml中的'base'里seed的值
    seed = args['base']['seed'] if opt.seed == -1 else opt.seed  
    setup_seed(seed)
    print("-------------------------------------------------------------------------")
    print("seed is fixed to {}".format(seed))
    print("-------------------------------------------------------------------------")

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])  #ckpt/mosi
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("-------------------------------------------------------------------------")
    print("ckpt root :", ckpt_root)
    print("-------------------------------------------------------------------------")
    print("***********------------------初始化主模型 models/P_RMF.py---------------***********")
    model = build_model(args).to(device)
    print("\033[1;35mTotal parameters: {}, Trainable parameters: {}\033[0m".format(*get_parameter_number(model)))
    # 加载完整数据 和 经过缺失处理的数据
    dataLoader = MMDataLoader(args)  #core/dataset.py
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

    #损失实例,包括gamma系数，sigma系数，kl系数，CE损失，MSE损失
    loss_fn = MultimodalLoss(args)
    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(args['dataset']['datasetName'])
    
    """✅ -----------------------------新增-----------------------------------------------------------"""
    # ====== 难度感知模块初始化 ======
    use_hardness = args['base'].get('use_hardness', False)
    hard_cfg = args.get('hardness', {})
    #curr_cfg = args.get('curriculum', {})

    if use_hardness:
        # 1) 难度估计器 初始化
        hardness_estimator = HardnessEstimator(
            train_mode=args['base'].get('train_mode', 'regression'),
            alpha_pred=hard_cfg.get('alpha_pred', 1.0),
            alpha_rec=hard_cfg.get('alpha_rec', 0.0),  # 想用重构误差时可以改成 0.3
            eps=hard_cfg.get('eps', 1e-6),
        ).to(device)

        # 2) 是否启用 loss 重加权 / 采样
        use_loss_reweight = hard_cfg.get('use_loss_reweight', True)
        use_sampling = hard_cfg.get('use_sampling', False)

        # # 3) 采样调度器（“软 curriculum”）
        # hardness_scheduler = HardnessScheduler(
        #     strategy=curr_cfg.get('strategy', 'none'),        # 'none' / 'hard' / 'easy'
        #     focus=curr_cfg.get('focus', 'hard'),
        #     start_epoch=curr_cfg.get('start_epoch', 2),
        #     warmup_epochs=curr_cfg.get('warmup_epochs', 5),
        #     max_strength=curr_cfg.get('max_strength', 0.5),
        #     beta=curr_cfg.get('beta', 2.0),
        #     eps=curr_cfg.get('eps', 1e-6),
        # )

        # 4) 全局 hardness 缓存：每个训练样本一个值
        train_dataset = dataLoader['train'].dataset
        num_train = len(train_dataset)
        global_hardness = torch.zeros(num_train, dtype=torch.float32)
    else:
        hardness_estimator = None
        hardness_scheduler = None
        global_hardness = None
        use_loss_reweight = False
        use_sampling = False
        train_dataset = dataLoader['train'].dataset 
    """✅ -----------------------------新增-------------------------------------------------------------"""

    print("----------------------------------------------开始训练！！！---------------------------------------------------------")

    for epoch in range(1, args['base']['n_epochs'] + 1):
        print(f'Training Epoch: {epoch}')
        start_time = time.time()

        # ====== 根据 hardness 构造本 epoch 的 train_loader（可选）======
        # if use_hardness and use_sampling and (hardness_scheduler is not None):
        #     # 使用上一轮的 global_hardness 生成采样权重
        #     sample_weights = hardness_scheduler.get_sample_weights(
        #         global_hardness=global_hardness.to(device),
        #         epoch=epoch,
        #     )  # [N]

        #     # 注意：这里用 replacement=False，保证每个样本每个 epoch 只被抽一次，
        #     # 这样不会破坏你在 MMDataset.__getitem__ 中基于 index==0 的缺失刷新逻辑。
        #     sampler = WeightedRandomSampler(
        #         weights=sample_weights,
        #         num_samples=len(sample_weights),
        #         replacement=False,
        #     )

        #     epoch_train_loader = DataLoader(
        #         train_dataset,
        #         batch_size=args['base']['batch_size'],
        #         sampler=sampler,
        #         num_workers=args['base']['num_workers'],
        #     )
        # else:
            # 不使用 hardness 采样时，保持原始行为
        
        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))

        train(model, train_loader, optimizer, loss_fn, epoch, metrics,
              hardness_estimator=hardness_estimator,
              global_hardness=global_hardness,
              use_hardness=use_hardness,
              use_loss_reweight=use_loss_reweight)

        # 每个 epoch 结束后，打印 hardness 分布方便你观察
        if use_hardness and (global_hardness is not None):
            hardness_diagnostics(global_hardness, epoch)
        

        if args['base']['do_validation']:
            valid_results = evaluate(model, dataLoader['valid'], loss_fn, epoch, metrics)
            best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root,
                                                  seed, 'valid', save_best_model=False)
            print(f'Current Best Valid Results: {best_valid_results}')

        test_results = evaluate(model, dataLoader['test'], loss_fn, epoch, metrics)
        best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed,
                                             'test', save_best_model=True)

        end_time = time.time()
        epoch_mins, epoch_secs = interval_time(start_time, end_time)
        print("Epoch: {}/{} | Current Best Test Results: {} | \n Time: {}m {}s".format(epoch, args['base']['n_epochs'],
                                                                                       best_test_results, epoch_mins,
                                                                                       epoch_secs))

        scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn, epoch, metrics,
         hardness_estimator=None, global_hardness=None,
         use_hardness=False, use_loss_reweight=False):
    """
    训练一个 epoch。
    如果 use_hardness=True：
      - 每个 batch 估计一次 hardness（不反向）
      - 用 hardness 做：
          1) global_hardness 的滑动更新（给下个 epoch 采样用）
          2) optional：对当前 batch 的情感预测 loss 做轻微重加权
    """
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (
            data['vision'].to(device),
            data['audio'].to(device),
            data['text'].to(device),
        )
        incomplete_input = (
            data['vision_m'].to(device),
            data['audio_m'].to(device),
            data['text_m'].to(device),
        )

        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}

        out = model(complete_input, incomplete_input)

        # ====== Step 1: 基于当前 batch 输出估计 hardness ======
        sample_weights = None
        if use_hardness and (hardness_estimator is not None):
            batch_hardness = hardness_estimator(
                preds=out['sentiment_preds'],
                labels=label['sentiment_labels'],
                rec_feats=out.get('rec_feats', None),
                complete_feats=out.get('complete_feats', None),
            )  # [B]

            if batch_hardness.dim() > 1:
                batch_hardness = batch_hardness.view(-1)

            # 1.1 更新全局 hardness（动量平滑）
            if global_hardness is not None:
                indices = data['index']           # [B]，Dataset 里返回的 index
                if isinstance(indices, torch.Tensor):
                    idx_cpu = indices.long().cpu()
                else:
                    idx_cpu = torch.tensor(indices, dtype=torch.long)

                old_values = global_hardness[idx_cpu]        # [B]
                new_values = batch_hardness.detach().cpu()   # [B]

                momentum = 0.8
                global_hardness[idx_cpu] = momentum * old_values + (1.0 - momentum) * new_values

            # 1.2 把 hardness 映射为 loss 权重（可选）
            if use_loss_reweight:

                # （0）最初始的方案： 简单线性映射：h∈[0,1] → w∈[0.5,1.5] (可选别的方式)
                #w = 0.5 + batch_hardness.to(device)    # [B]
                # 再归一化到均值为 1，避免改变 overall scale
                #w = w / (w.mean() + 1e-8)
                #sample_weights = w
                sample_weights = map_hardness_to_weights(batch_hardness, mode='piecewise', alpha= 1.0, device=device)

        # ====== Step 2: 计算 loss（支持样本级加权） ======
        loss = loss_fn(out, label, sample_weights=sample_weights)

        # ====== Step 3: 反向传播 & 更新参数 ======
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        # ====== 记录预测与标签，用于计算指标 ======
        y_pred.append(out['sentiment_preds'].detach().cpu())
        y_true.append(label['sentiment_labels'].detach().cpu())

        # ====== 记录各项 loss，方便打印 ======
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

    #{'Has0_acc_2': , 'Has0_F1_score': , 'Non0_acc_2': , 'Non0_F1_score': , 'Mult_acc_5': , 'Mult_acc_7': , 'MAE': , 'Corr': }
    loss_dict = {key: value / (cur_iter + 1) for key, value in loss_dict.items()}
    #{'loss': , 'l_sp': , 'l_rec': , 'l_kl': }
    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')


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
