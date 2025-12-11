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
    print("----------------------------------------------开始训练！！！---------------------------------------------------------")
    for epoch in range(1, args['base']['n_epochs'] + 1):
        print(f'Training Epoch: {epoch}')
        start_time = time.time()
        train_loader = tqdm(dataLoader['train'], total=len(dataLoader['train']))

        #------------------------------！！！进入 train   ！！！--------------------------------
        train(model, train_loader, optimizer, loss_fn, epoch, metrics)
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


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    y_pred, y_true = [], []
    loss_dict = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):#读取 train_loader 会调用 dataset中的__getitem__函数,读够一个batch（16）的数据就开始往下走
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device) #(16,1)
        label = {'sentiment_labels': sentiment_labels}
        #******---------------------------！！！开始前向传播：进入P-RMFmodel的forward函数 ！！！--------------------******
        out = model(complete_input, incomplete_input)
        # 返回值out字典：{'sentiment_preds':预测值,'rec_feats':可用于重建损失的重建特征或None,
        #                'complete_feats':对齐监督的完整特征或None, 'kl_loss':PMG的KL/重建混合项}
        loss = loss_fn(out, label)

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                loss_dict[key] = value.item()
        else:
            for key, value in loss.items():
                loss_dict[key] += value.item()

    pred, true = torch.cat(y_pred), torch.cat(y_true) #[1284,1]
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
