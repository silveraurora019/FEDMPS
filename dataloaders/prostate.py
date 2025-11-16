import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import logging
import math

class ProstateStaticDataset(Dataset):
    """
    用于联邦学习的静态前列腺数据集加载器。
    
    假定数据已按以下结构预处理 (由 preprocess_prostate_nifti.py 生成):
    - base_path/
      - site1/
        - image_static/
          - slice_00001.pt
          - slice_00002.pt
        - mask_static/
          - slice_00001.pt
          - slice_00002.pt
      - site2/
        ...
    """
    
    def __init__(self, client_idx, base_path, split='train', transform=None):
        self.transform = transform  # 用于运行时的额外增强 (例如翻转)
        self.split = split
        self.client_names = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6']
        client_name = self.client_names[client_idx]

        # base_path 现在指向预处理的静态文件根目录
        # 注意：我们使用 'image_static' 目录来匹配预处理脚本的输出
        image_dir = os.path.join(base_path, client_name, 'image_static')
        
        # Glob 查找所有预处理好的 .pt 图像文件
        image_files = sorted(glob(os.path.join(image_dir, "*.pt")))
        
        logging.info(f"Client {client_idx} ({client_name}) found {len(image_files)} static slices in {image_dir}.")

        if not image_files:
             logging.warning(f"Client {client_idx} ({client_name}) - No static files found in {image_dir}.")

        # 使用与 NIfTI 加载器相同的种子进行可复现的打乱
        random.Random(42).shuffle(image_files)

        # 6:2:2 划分 (与 NIfTI 加载器逻辑相同)
        total_slices = len(image_files)
        train_idx = math.floor(total_slices * 0.6)
        val_idx = math.floor(total_slices * 0.8)
        
        if split == 'train':
            self.image_list = image_files[:train_idx]
        elif split == 'val':
            self.image_list = image_files[train_idx:val_idx]
        elif split == 'test':
            self.image_list = image_files[val_idx:]
            
        logging.info(f"Client {client_idx} ({client_name}) using split '{split}'. Slices: {len(self.image_list)}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if not self.image_list or idx >= len(self.image_list):
             raise IndexError(f"Index {idx} out of bounds for static slice_pool of size {len(self.image_list)}")

        img_path = self.image_list[idx]
        
        # 从图像路径推导掩码路径
        mask_path = img_path.replace(os.sep + 'image_static' + os.sep, os.sep + 'mask_static' + os.sep)
        
        try:
            # 动态加载：现在只加载两个小文件，而不是整个 3D NIfTI
            image_tensor = torch.load(img_path)  # (3, H, W) Float
            mask_tensor = torch.load(mask_path)  # (1, H, W) Long

            # (可选) 在此处应用运行时的 transform (例如翻转、旋转)
            if self.transform:
                # 注意：您需要确保您的 transform (例如 v2) 可以正确处理 (img, mask)
                # image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
                pass
            
            # 损失函数 (loss.py) 期望掩码是 Float 类型 (用于内部 one-hot)
            #
            return image_tensor, mask_tensor.float()
        
        except Exception as e:
            logging.error(f"Error loading static file {img_path} or {mask_path}: {e}")
            # 返回匹配预处理格式的空张量
            return torch.zeros((3, 384, 384)), torch.zeros((1, 384, 384)).float()
