# 文件名: dataloaders/prostate.py
# (已修复维度不匹配的错误)

import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import random
from torchvision.transforms import v2 as transforms
import re 

class Prostate(Dataset):
    """ 
    加载 2D 预处理前列腺切片 (sample*.npy) 的数据集类。
    在 __init__ 中实现 60:20:20 (Train:Val:Test) 划分。
    """
    
    def __init__(self, client_idx=None, base_path=None, split='train', transform=None):
        self.root_dir = base_path 
        self.split = split
        
        # 1. 训练集几何增强 (同步应用到图像和掩码)
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        
        # 2. 仅用于图像的颜色增强
        self.image_only_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ])

        # 客户端名称列表
        self.client_name = ['Site1', 'Site2', 'Site3', 'Site4', 'Site5', 'Site6'] 
        
        image_dir = os.path.join(self.root_dir, self.client_name[client_idx], 'image')
        self.image_files = glob(os.path.join(image_dir, 'sample*.npy'))
        
        # 按数字顺序排序文件
        def get_num(f):
            match = re.search(r'sample(\d+)\.npy', os.path.basename(f))
            return int(match.group(1)) if match else -1
        
        self.image_files.sort(key=get_num)
        
        # 使用固定种子打乱列表
        random.Random(42).shuffle(self.image_files)
        
        # 按照 60:20:20 比例划分索引
        total_len = len(self.image_files)
        train_len = int(total_len * 0.6)
        val_len = int(total_len * 0.2)

        if split == 'train':
            self.image_list = self.image_files[:train_len]
        elif split == 'val':
            self.image_list = self.image_files[train_len : train_len + val_len]
        elif split == 'test':
            self.image_list = self.image_files[train_len + val_len:]
        else:
            raise ValueError(f"无效的 split 名称: {split}")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        mask_path = img_path.replace(os.sep + 'image' + os.sep, os.sep + 'mask' + os.sep)
        
        try:
            # --- 关键修改 1: 加载数据 ---
            # 假设 image 是 (H, W, 3) 
            # 假设 mask 是 (H, W)
            image_np = np.load(img_path, allow_pickle=True) 
            mask_np = np.load(mask_path, allow_pickle=True)
            
            # --- 关键修改 2: 转换为正确的 3D 张量形状 ---
            # image: (H, W, 3) -> (3, H, W)
            image = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
            # mask: (H, W) -> (1, H, W)
            mask = torch.from_numpy(mask_np).long().unsqueeze(0) 

        except Exception as e:
            print(f"加载文件时出错 {img_path} 或 {mask_path}: {e}")
            return torch.zeros((3, 384, 384)), torch.zeros((1, 384, 384)).long()

        
        if self.split == 'train':
            # --- 关键修改 3: 调整堆叠和分离的逻辑 ---
            # 1. 堆叠 (3, H, W) 和 (1, H, W) -> (4, H, W)
            stacked = torch.cat((image, mask.float()), dim=0) 
            
            # 2. 应用几何变换
            stacked_aug = self.train_transform(stacked)
            
            # 3. 分离: 图像是前3个通道, 掩码是第4个通道
            image = stacked_aug[0:3, :, :]
            mask = stacked_aug[3:4, :, :]
            
            # 4. 仅对图像应用颜色增强
            image = self.image_only_transform(image)
        
        # (对于 'val' 和 'test', 图像已经是 (3, H, W)，掩码是 (1, H, W)，无需操作)

        # 确保掩码是 Long 类型
        mask = mask.long()
        
        return image, mask


# # 文件名: dataloaders/prostate.py
# # (这是一个新文件)

# import os
# import torch
# import numpy as np
# from glob import glob
# from torch.utils.data import Dataset
# import random
# from torchvision.transforms import v2 as transforms
# import re # 用于数字排序

# class Prostate(Dataset):
#     """ 
#     加载 2D 预处理前列腺切片 (sample*.npy) 的数据集类。
#     在 __init__ 中实现 60:20:20 (Train:Val:Test) 划分。
#     """
    
#     def __init__(self, client_idx=None, base_path=None, split='train', transform=None):
#         self.root_dir = base_path 
#         self.split = split
        
#         # --- 定义数据增强 (防止过拟合) ---
#         # 1. 训练集几何增强 (同步应用到图像和掩码)
#         self.train_transform = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.5),
#             transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
#         ])
        
#         # 2. 仅用于图像的颜色增强
#         self.image_only_transform = transforms.Compose([
#             transforms.ColorJitter(brightness=0.3, contrast=0.3),
#         ])
#         # ---------------------------

#         # 客户端名称列表
#         self.client_name = ['Site1', 'Site2', 'Site3', 'Site4', 'Site5', 'Site6'] 
        
#         image_dir = os.path.join(self.root_dir, self.client_name[client_idx], 'image')
#         self.image_files = glob(os.path.join(image_dir, 'sample*.npy'))
        
#         # --- 关键步骤: 按数字顺序排序文件 ---
#         # (确保 sample10.npy 在 sample2.npy 之后)
#         def get_num(f):
#             match = re.search(r'sample(\d+)\.npy', os.path.basename(f))
#             return int(match.group(1)) if match else -1
        
#         self.image_files.sort(key=get_num)
#         # -------------------------------------
        
#         # 使用固定种子打乱列表，以保证划分的可复现性
#         random.Random(42).shuffle(self.image_files)
        
#         # 按照 60:20:20 比例划分索引
#         total_len = len(self.image_files)
#         train_len = int(total_len * 0.6)
#         val_len = int(total_len * 0.2)
#         # test_len = total_len - train_len - val_len

#         if split == 'train':
#             self.image_list = self.image_files[:train_len]
#         elif split == 'val':
#             self.image_list = self.image_files[train_len : train_len + val_len]
#         elif split == 'test':
#             self.image_list = self.image_files[train_len + val_len:]
#         else:
#             raise ValueError(f"无效的 split 名称: {split}")

#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         img_path = self.image_list[idx]
        
#         # 从图像路径推导掩码路径
#         # 例如: .../Site1/image/sample0.npy -> .../Site1/mask/sample0.npy
#         mask_path = img_path.replace(os.sep + 'image' + os.sep, os.sep + 'mask' + os.sep)
        
#         try:
#             image = np.load(img_path, allow_pickle=True) # (H, W)
#             mask = np.load(mask_path, allow_pickle=True) # (H, W)
#         except Exception as e:
#             print(f"加载文件时出错 {img_path} 或 {mask_path}: {e}")
#             return torch.zeros((3, 384, 384)), torch.zeros((1, 384, 384)).long()

        
#         # 转换为 PyTorch Tensors
#         image = torch.from_numpy(image).unsqueeze(0).float() # (1, H, W)
#         mask = torch.from_numpy(mask).long().unsqueeze(0)   # (1, H, W)

#         # --- 应用数据增强 ---
#         if self.split == 'train':
#             # 1. 堆叠图像和掩码，以进行同步的几何变换
#             stacked = torch.cat((image, mask.float()), dim=0) # (2, H, W)
#             stacked_aug = self.train_transform(stacked)
#             image, mask = stacked_aug[0:1, :, :], stacked_aug[1:2, :, :]
            
#             # 2. 将图像复制为3通道 (以应用ColorJitter)
#             image = image.repeat(3, 1, 1) # (3, H, W)
#             # 3. 仅对图像应用颜色增强
#             image = self.image_only_transform(image)
#         else:
#             # 对于 val/test, 只需复制通道以匹配模型输入
#             image = image.repeat(3, 1, 1) # (3, H, W)
#         # ---------------------------

#         # 确保掩码是 Long 类型，并且形状为 (B, 1, H, W)
#         mask = mask.long()
        
#         return image, mask