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
        h = hardness.to(device)
        # 预热期返回全1
        if epoch <= self.warmup_epochs:
            return torch.ones_like(h)

        # 进度控制：覆盖80%的训练周期
        effective_epochs = self.total_epochs - self.warmup_epochs
        current_step = epoch - self.warmup_epochs
        progress = np.clip(current_step / (effective_epochs * 0.8 + 1e-6), 0.0, 1.0)

        # [优化] 关注中心 mu: 0.3 -> 0.75 (留出顶部空间)
        mu = 0.3 + 0.45 * progress  
        
        sigma = 0.2 - 0.05 * progress
        strength = 1.0 + 2.0 * progress
        
        # [优化] 基础权重衰减：1.0 -> 0.5
        base_weight = 1.0 - 0.5 * progress

        # 高斯加权
        gaussian_term = torch.exp(-0.5 * ((h - mu) ** 2) / (sigma ** 2))
        weights = base_weight + strength * gaussian_term

        # [优化] 软熔断机制：提高阈值到 0.9
        soft_start = 0.90
        soft_end = 0.98 # 留一点余地，防止把 0.96 的样本全杀了
        
        outlier_mask = h > soft_start
        if outlier_mask.any():
            decay_factor = torch.clamp(
                1.0 - (h[outlier_mask] - soft_start) / (soft_end - soft_start), 
                min=0.0, max=1.0
            )
            weights[outlier_mask] *= decay_factor
            weights[outlier_mask] += 0.05 # 兜底

        # [移除] 移除此处的均值归一化，交给 loss 层处理
        # if weights.sum() > 1e-6:
        #     weights = weights / weights.mean()
        
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
         use_hardness=False, use_loss_reweight=False, 
         weight_scheduler=None, hardness_momentum=0.9): # [新增] 动量参数
    """
    训练一个 epoch。
    改进点：
      1. 修复 global_hardness 冷启动问题
      2. 优化权重获取逻辑的鲁棒性
    """
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        # 数据加载
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

        # 前向传播
        out = model(complete_input, incomplete_input)

        # ====== Step 1: 难度感知与权重计算 ======
        sample_weights = None
        
        if use_hardness and (hardness_estimator is not None):
            # 1.1 计算当前 Batch 的瞬时难度
            batch_hardness = hardness_estimator(
                preds=out['sentiment_preds'],
                labels=label['sentiment_labels'],
                rec_feats=out.get('rec_feats', None),
                complete_feats=out.get('complete_feats', None),
            )  # [B]

            if batch_hardness.dim() > 1:
                batch_hardness = batch_hardness.view(-1)

            # 1.2 更新全局 Hardness (动量平滑)
            if global_hardness is not None:
                indices = data['index']
                if isinstance(indices, torch.Tensor):
                    idx_cpu = indices.long().cpu()
                else:
                    idx_cpu = torch.tensor(indices, dtype=torch.long)

                # [修复] 冷启动逻辑：第1个epoch直接赋值，避免0值污染；后续epoch进行EMA更新
                if epoch == 1:
                    global_hardness[idx_cpu] = batch_hardness.detach().cpu()
                else:
                    old_values = global_hardness[idx_cpu]
                    new_values = batch_hardness.detach().cpu()
                    global_hardness[idx_cpu] = hardness_momentum * old_values + (1.0 - hardness_momentum) * new_values

            # 1.3 获取用于加权的 Hardness
            # [优化] 设置默认值，防止分支逻辑导致 current_h 未定义
            current_h = batch_hardness 

            if use_loss_reweight and weight_scheduler is not None:
                # 优先读取平滑后的全局难度 (更稳定)
                if global_hardness is not None:
                    indices = data['index']
                    if isinstance(indices, torch.Tensor):
                        idx_cpu = indices.long().cpu()
                    else:
                        idx_cpu = torch.tensor(indices, dtype=torch.long)
                    current_h = global_hardness[idx_cpu].to(device)
                
                # 调用调度器获取权重
                sample_weights = weight_scheduler.get_weights(current_h, epoch, device)

        # ====== Step 2: 计算 Loss (支持样本级加权) ======
        loss = loss_fn(out, label, sample_weights=sample_weights)

        # ====== Step 3: 反向传播 & 更新 ======
        optimizer.zero_grad()
        loss['loss'].backward()
        optimizer.step()

        # ====== 记录数据 ======
        y_pred.append(out['sentiment_preds'].detach().cpu())
        y_true.append(label['sentiment_labels'].detach().cpu())

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

    # ====== Epoch 结束后的统计 ======
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    loss_dict = {key: value / (cur_iter + 1) for key, value in loss_dict.items()}
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
