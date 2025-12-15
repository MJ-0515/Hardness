import os
import torch
import yaml
import argparse
import numpy as np
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

class DynamicWeightScheduler:
    def __init__(self, total_epochs, warmup_epochs=5):
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def get_weights(self, hardness, epoch, device='cuda'):
        """
        根据训练进度动态调整关注点：
        - 初期：关注中低难度 (打基础)
        - 中后期：关注高难度 (提上限)
        - 全程：抑制极值离群点 (防崩塌)
        """
        h = hardness.to(device)
        weights = torch.ones_like(h)
        
        # --- 1. 预热期 (Warmup) ---
        # 前几轮不加权，让 VAE 和 GRL 充分预热
        if epoch < self.warmup_epochs:
            return weights

        # --- 2. 计算进度因子 (Progress) ---
        # 将剩余的轮数映射到 [0, 1]
        # 我们设定在 80% 的时候达到最难的关注点，最后 20% 保持稳定
        effective_epochs = self.total_epochs - self.warmup_epochs
        current_step = epoch - self.warmup_epochs
        progress = np.clip(current_step / (effective_epochs * 0.4 + 1e-6), 0.0, 1.0)

        # --- 3. 动态移动关注中心 (Moving Focus) ---
        # 核心逻辑：关注中心 mu 从 0.3 (易) 逐渐滑向 0.8 (难)
        # 这样 Epoch 10-25 学简单的，Epoch 40+ 强迫模型去学难的
        mu = 0.3 + 0.5 * progress  
        
        # 聚焦范围 sigma：随着训练进行，聚焦越来越集中 (0.2 -> 0.15)
        sigma = 0.2 - 0.05 * progress
        
        # 加权强度 strength：后期强度更大 (1.0 -> 3.0)
        strength = 1.0 + 2.0 * progress

        # --- 4. 高斯加权公式 ---
        # w = 1 + strength * exp(...)
        gaussian_weight = torch.exp(-0.5 * ((h - mu) ** 2) / (sigma ** 2))
        weights = 1.0 + strength * gaussian_weight

        # --- 5. 离群点熔断 (Outlier Cutoff) ---
        # 始终抑制难度极高 (>0.95) 的样本，防止脏数据破坏模型
        # 注意：这里的阈值要比 mu 的终点(0.8)高，给难样本留出空间
        outlier_thresh = 0.95
        mask_outlier = h > outlier_thresh
        if mask_outlier.any():
            # 对离群点降权到 0.1 (保留一点点梯度，或者直接设为0)
            weights[mask_outlier] = 0.1

        # --- 6. 均值归一化 ---
        # 保持 batch 总 loss 规模稳定
        if weights.sum() > 1e-6:
            weights = weights / weights.mean()
        
        return weights

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


    # [新增] 初始化动态权重调度器
    weight_scheduler = DynamicWeightScheduler(
        total_epochs=args['base']['n_epochs'], 
        warmup_epochs=5  # 前5轮预热
    )

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
        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))

        train(model, train_loader, optimizer, loss_fn, epoch, metrics,
              hardness_estimator=hardness_estimator,
              global_hardness=global_hardness,
              use_hardness=use_hardness,
              use_loss_reweight=use_loss_reweight,
              weight_scheduler=weight_scheduler)

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
         use_hardness=False, use_loss_reweight=False, weight_scheduler=None):
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

                momentum = 0.9
                global_hardness[idx_cpu] = momentum * old_values + (1.0 - momentum) * new_values

            # 1.2 把 hardness 映射为 loss 权重（可选）
            if use_loss_reweight and weight_scheduler is not None:
               # [改进] 尝试从 global_hardness 获取当前 batch 的平滑难度
               if global_hardness is not None:
                 indices = data['index']
                 if isinstance(indices, torch.Tensor):
                    idx_cpu = indices.long().cpu()
                 else:
                    idx_cpu = torch.tensor(indices, dtype=torch.long)
                
                 # 使用历史平滑难度 (更稳定)
                 current_h = global_hardness[idx_cpu].to(device)
            else:
                # 降级方案
               current_h = batch_hardness

            sample_weights = weight_scheduler.get_weights(current_h, epoch, device)
            #（0）最初始的方案： 简单线性映射：h∈[0,1] → w∈[0.5,1.5] ##3
            #w = 0.5 + batch_h_smooth.to(device)    # [B]
            #再归一化到均值为 1，避免改变 overall scale
            #w = w / (w.mean() + 1e-8)
            #sample_weights = w
            ####
            #sample_weights = map_hardness_to_weights(current_h, mode='piecewise', alpha= 1.0, device=device)

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
