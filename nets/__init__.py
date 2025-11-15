# 文件名: new_nets_init.py

import logging
import copy
from .unet import UNet,UNet_pro # <-- 导入新的 unet 文件

def build_model(args, clients, device):

    n_classes = 2

    if args.dataset=='pmri': 
        n_classes = 3 
    elif args.dataset == 'fundus':
        n_classes = 3 
    elif args.dataset == 'prostate':
        n_classes = 2 
        
    if args.model == 'unet':
        model = UNet(out_channels=n_classes, dropout_p=args.dropout_rate)
    elif args.model == 'unet_pro':
        # 确保将 dropout_rate 传递给 UNet_pro
        model = UNet_pro(out_channels=n_classes, dropout_p=args.dropout_rate)
    else:
        logging.error('unknow model')
        raise ValueError(f"Unknown model type: {args.model}")

    model = model.to(device)

    local_models = []
    for id, c in enumerate(clients):
        local_models.append(copy.deepcopy(model))

    total_params = sum(p.numel() for p in model.parameters())
    logging.info('Model {} parameters: {} M'.format(args.model, total_params/1024/1024))

    global_model = copy.deepcopy(model)
    return local_models, global_model