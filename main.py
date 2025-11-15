# 文件名: main.py (集成 MPS 版本)

import logging
import torch
import os
import numpy as np
import random
import argparse
import copy
from pathlib import Path

from utils import set_for_logger
from dataloaders import build_dataloader
# 确保使用您之前修复过 bug 的 loss.py
from loss import DiceLoss, JointLoss
import torch.nn.functional as F
from nets import build_model

# --- 新增：导入 MPS 聚合器 ---
from aggregator_mps import MPSAggregator


@torch.no_grad()
def get_client_logits_targets(local_models, dataloaders, device):
    """
    从所有客户端提取 Logits 和 Targets 用于 MPS 计算。
    
    注意：医学图像分割输出为 (B, C, H, W)。
    为了计算 Margin Profile，我们需要将其展平为 (N, C)，其中 N 是像素总数。
    为了节省内存和计算时间，我们会先对特征图进行下采样。
    """
    all_logits_list = []
    all_targets_list = []
    
    # 下采样尺寸，用于减小计算量 (例如将 384x384 降为 64x64)
    # 这保留了分布特征，同时大幅降低内存占用
    downsample_size = (64, 64) 

    for model, loader in zip(local_models, dataloaders):
        model.eval()
        client_logits = []
        client_targets = []
        
        try:
            # 为了速度，我们不需要遍历整个验证集，取一部分即可
            # 如果显存足够，可以遍历全部
            max_batches = 10 
            batch_count = 0
            
            for x, target in loader:
                if batch_count >= max_batches: break
                
                x = x.to(device)
                target = target.to(device)
                
                # 1. 模型前向传播
                # UNet_pro 返回 (output, z, shadow)，我们只需要 output (logits)
                out = model(x)
                if isinstance(out, tuple):
                    logits = out[0]
                else:
                    logits = out
                
                # 2. 下采样 Logits: (B, C, H, W) -> (B, C, h, w)
                logits_down = F.interpolate(logits, size=downsample_size, mode='bilinear', align_corners=False)
                
                # 3. 下采样 Targets: (B, 1, H, W) or (B, H, W) -> (B, h, w)
                if target.dim() == 3:
                    target = target.unsqueeze(1) # (B, 1, H, W)
                target_down = F.interpolate(target.float(), size=downsample_size, mode='nearest').long().squeeze(1)
                
                # 4. 展平并存储
                # logits: (B, C, h, w) -> (B, h, w, C) -> (-1, C)
                logits_flat = logits_down.permute(0, 2, 3, 1).reshape(-1, logits_down.shape[1])
                # target: (B, h, w) -> (-1)
                target_flat = target_down.reshape(-1)
                
                client_logits.append(logits_flat.cpu())
                client_targets.append(target_flat.cpu())
                
                batch_count += 1
            
            if len(client_logits) > 0:
                # 拼接该客户端所有 batch 的像素点
                full_logits = torch.cat(client_logits, dim=0)
                full_targets = torch.cat(client_targets, dim=0)
                
                all_logits_list.append(full_logits)
                all_targets_list.append(full_targets)
            else:
                logging.warning("客户端验证集为空。")
                all_logits_list.append(None)
                all_targets_list.append(None)

        except Exception as e:
             logging.error(f"提取 Logits/Targets 时出错: {e}")
             all_logits_list.append(None)
             all_targets_list.append(None)
    
    return all_logits_list, all_targets_list


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--data_root', type=str, required=False, 
                        default="/data/myn/dataset/Fundus", 
                        help="Data directory")
    parser.add_argument('--dataset', type=str, default='fundus', 
                        help="Dataset type: 'fundus' or 'prostate'")
    
    parser.add_argument('--model', type=str, default='unet_pro', help='Model type')

    parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:1', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--experiment', type=str, default='experiment_mps', help='Experiment name')

    parser.add_argument('--test_step', type=int, default=1)
    
    # --- 新增：MPS 参数 ---
    parser.add_argument('--mps_gamma', type=float, default=1.0, help='Gamma for MPS similarity (exp(-gamma * dist))')
    parser.add_argument('--mps_temp', type=float, default=0.1, help='Temperature for softmax weighting in MPS')
    parser.add_argument('--sim_start_round', type=int, default=0, help='Round to start using MPS aggregation')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate for the model')

    args = parser.parse_args()
    return args

def communication(server_model, models, client_weights):
    with torch.no_grad():
        device = next(server_model.parameters()).device
        if not isinstance(client_weights, torch.Tensor):
            client_weights = torch.tensor(client_weights, dtype=torch.float32, device=device)
        else:
            client_weights = client_weights.to(device)
            
        # 归一化权重以防万一
        client_weights = client_weights / client_weights.sum()
        
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32, device=device)
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key].to(device)
            server_model.state_dict()[key].data.copy_(temp)
    return server_model

def train(cid, model, dataloader, device, optimizer, epochs, loss_func):
    model.train()
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    for epoch in range(epochs):
        train_acc = 0.
        loss_all = 0.
        if len(dataloader) == 0:
            continue
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            
            if is_unet_pro:
                output, _, _ = model(x) 
            else:
                output = model(x)
                
            optimizer.zero_grad()
            loss = loss_func(output, target)
            loss_all += loss.item()
            train_acc += DiceLoss().dice_coef(output, target).item()
            loss.backward()
            optimizer.step()
        
        if len(dataloader) > 0:
            avg_loss = loss_all / len(dataloader)
            train_acc = train_acc / len(dataloader)
            logging.info('Client: [%d]  Epoch: [%d]  train_loss: %f train_acc: %f'%(cid, epoch, avg_loss, train_acc))

def test(model, dataloader, device, loss_func):
    model.eval()
    loss_all = 0
    test_acc = 0
    is_unet_pro = model.__class__.__name__ == 'UNet_pro'
    
    if len(dataloader) == 0:
        return 0.0, 0.0

    with torch.no_grad():
        for x, target in dataloader:
            x = x.to(device)
            target = target.to(device)
            if is_unet_pro:
                output, _, _ = model(x)
            else:
                output = model(x)
            loss = loss_func(output, target)
            loss_all += loss.item()
            test_acc += DiceLoss().dice_coef(output, target).item()
        
    acc = test_acc / len(dataloader)
    loss = loss_all / len(dataloader)
    return loss, acc

def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # 2. 动态定义客户端列表
    if args.dataset == 'fundus':
        clients = ['site1', 'site2', 'site3', 'site4']
    elif args.dataset == 'prostate':
        clients = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6']
    else:
        raise ValueError(f"Unknown client list for dataset: {args.dataset}")

    # 3. build dataset
    train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)
    client_weight_tensor = torch.tensor(client_weight, dtype=torch.float32, device=device)

    # 4. build model
    local_models, global_model = build_model(args, clients, device)

    # --- 新增：初始化 MPS Aggregator ---
    mps_aggregator = MPSAggregator(
        device=device,
        gamma=args.mps_gamma,
        temperature=args.mps_temp
    )
    logging.info(f"MPS Aggregator initialized. Gamma={args.mps_gamma}, Temp={args.mps_temp}")
    # --- 新增结束 ---

    loss_fun = JointLoss() 
    
    optimizer = []
    for id in range(len(clients)):
        optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

    best_dice = 0
    best_dice_round = 0
    best_local_dice = []

    weight_save_dir = os.path.join(args.save_dir, args.experiment)
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    
    for r in range(args.rounds):
        logging.info('-------- Commnication Round: %3d --------'%r)

        # 1. 本地训练
        for idx, client in enumerate(clients):
            train(idx, local_models[idx], train_dls[idx], device, optimizer[idx], args.epochs, loss_fun)
            
        temp_locals = copy.deepcopy(local_models)
        
        # --- 修改：聚合逻辑 (MPS) ---
        if r >= args.sim_start_round:
            logging.info('Calculating Margin Profile Similarity (MPS)...')
            
            # 3a. 提取 Logits 和 Targets (使用验证集)
            logits_list, targets_list = get_client_logits_targets(temp_locals, val_dls, device)
            
            # 3b. 计算 MPS 权重
            mps_weights = mps_aggregator.compute_weights(logits_list, targets_list)
            
            if mps_weights is not None:
                logging.info(f'MPS Weights: {mps_weights.cpu().numpy()}')
                aggr_weights = mps_weights
            else:
                logging.warning("MPS weights computation failed. Fallback to FedAvg.")
                aggr_weights = client_weight_tensor
            
            # 3c. 执行聚合
            communication(global_model, temp_locals, aggr_weights)

        else: 
            # 3d. 早期轮次使用 FedAvg
            logging.info('Using standard FedAvg aggregation.')
            communication(global_model, temp_locals, client_weight_tensor)
        # --- 修改结束 ---


        # 4. 分发全局模型
        global_w = global_model.state_dict()
        for idx, client in enumerate(clients):
            local_models[idx].load_state_dict(global_w)


        if r % args.test_step == 0:
            # 5. 测试
            avg_loss = []
            avg_dice = []
            for idx, client in enumerate(clients):
                loss, dice = test(local_models[idx], test_dls[idx], device, loss_fun)
                logging.info('client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
                avg_dice.append(dice)
                avg_loss.append(loss)

            avg_dice_v = sum(avg_dice) / len(avg_dice) if len(avg_dice) > 0 else 0
            avg_loss_v = sum(avg_loss) / len(avg_loss) if len(avg_loss) > 0 else 0
            
            logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))

            # 7. 保存最佳模型
            if best_dice < avg_dice_v:
                best_dice = avg_dice_v
                best_dice_round = r
                best_local_dice = avg_dice

                weight_save_path = os.path.join(weight_save_dir, 'best.pth')
                torch.save(global_model.state_dict(), weight_save_path)
            

    logging.info('-------- Training complete --------')
    logging.info('Best avg dice score %f at round %d '%( best_dice, best_dice_round))
    for idx, client in enumerate(clients):
        logging.info('client: %s  test_acc:  %f '%(client, best_local_dice[idx] if idx < len(best_local_dice) else 0.0))


if __name__ == '__main__':
    args = get_args()
    main(args)