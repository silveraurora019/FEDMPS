# dataloaders/prostate_nifti_dataset.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
from glob import glob
import random
import logging
import math

class ProstateNIFTIDataset(Dataset):
    """
    ... (Docstring 不变) ...
    """
    def __init__(self, client_idx, base_path, split='train', transform=None):
        self.transform = transform  # <--- transform 现在会被 __init__ 接收
        self.split = split
        # 假设您的 6 个站点名为 'site1' 到 'site6'
        self.client_names = ['site1', 'site2', 'site3', 'site4', 'site5', 'site6']
        client_name = self.client_names[client_idx]

        # 假设数据结构为: base_path/site1/Case00_segmentation.nii.gz
        data_dir = os.path.join(base_path, client_name)
        
        # 使用 glob 查找该客户端的所有分割文件
        seg_paths = sorted(glob(os.path.join(data_dir, "*_segmentation.nii.gz")))
        
        logging.info(f"Client {client_idx} ({client_name}) found {len(seg_paths)} 3D volumes in {data_dir}.")

        # 存储所有有效切片的 (img_path, label_path, slice_idx)
        self.slice_pool = []
        self._read_slices_into_memory(seg_paths) # 传入 seg_paths 列表

        # 对所有切片进行一次性随机打乱（使用固定种子保证可复现性）
        random.Random(42).shuffle(self.slice_pool)

        # 划分 6:2:2
        total_slices = len(self.slice_pool)
        train_idx = math.floor(total_slices * 0.6) # 60%
        val_idx = math.floor(total_slices * 0.8)   # 60% + 20% = 80%
        
        if split == 'train':
            self.slice_pool = self.slice_pool[:train_idx]
            logging.info(f"Client {client_idx} ({client_name}) using split 'train'. Slices: {len(self.slice_pool)}")
        elif split == 'val': # 验证集
            self.slice_pool = self.slice_pool[train_idx:val_idx]
            logging.info(f"Client {client_idx} ({client_name}) using split 'val'. Slices: {len(self.slice_pool)}")
        elif split == 'test': # 测试集
            self.slice_pool = self.slice_pool[val_idx:]
            logging.info(f"Client {client_idx} ({client_name}) using split 'test'. Slices: {len(self.slice_pool)}")


    def _read_slices_into_memory(self, seg_paths):
        """
        填充 self.slice_pool
        """
        for seg_path in seg_paths:
            img_path = seg_path.replace("_segmentation.nii.gz", ".nii.gz")
            if not os.path.exists(img_path):
                logging.warning(f"Image file not found for {seg_path}, skipping.")
                continue
                
            try:
                label_sitk = sitk.ReadImage(seg_path)
                label_npy = sitk.GetArrayFromImage(label_sitk)
                
                for slice_idx in range(label_npy.shape[0]):
                    if label_npy[slice_idx, :, :].max() > 0:
                        self.slice_pool.append((img_path, seg_path, slice_idx))
            except Exception as e:
                logging.error(f"Error loading or processing {seg_path}: {e}")
        
        logging.info(f"Client found {len(self.slice_pool)} total valid slices before splitting.")

    def __len__(self):
        return len(self.slice_pool)

    def _preprocess_slice(self, x):
        """
        应用 aegis 强度裁剪
        """
        mask = x > 0
        y = x[mask]

        if y.shape[0] > 0:
            lower = np.percentile(y, 0.2)
            upper = np.percentile(y, 99.8)
            x[mask & (x < lower)] = lower
            x[mask & (x > upper)] = upper
        return x

    def __getitem__(self, idx):
        if not self.slice_pool or idx >= len(self.slice_pool):
             raise IndexError(f"Index {idx} out of bounds for slice_pool of size {len(self.slice_pool)}")
        
        img_path, label_path, slice_idx = self.slice_pool[idx]
        
        try:
            # ... (加载 image_slice 和 mask_slice 的代码不变) ...
            img_sitk = sitk.ReadImage(img_path)
            img_npy_3d = sitk.GetArrayFromImage(img_sitk)
            image_slice = img_npy_3d[slice_idx, :, :].astype(np.float32)
            
            label_sitk = sitk.ReadImage(label_path)
            label_npy_3d = sitk.GetArrayFromImage(label_sitk)
            mask_slice = label_npy_3d[slice_idx, :, :].astype(np.uint8)

            # 1. Aegis 预处理: 强度裁剪
            image_slice = self._preprocess_slice(image_slice)

            # 2. Aegis 预处理: Z-score 标准化
            mean = image_slice.mean()
            std = image_slice.std()
            if std > 0:
                image_slice = (image_slice - mean) / std
            
            # 3. 适配模型: 1 通道 -> 3 通道
            image = np.repeat(image_slice[np.newaxis, :, :], 3, axis=0)
            
            # 4. 格式化掩码
            mask_slice[mask_slice > 1] = 1 # 确保为 0 或 1
            mask = mask_slice[np.newaxis, :, :] # 增加通道维度 (1, H, W)
            
            image_tensor = torch.from_numpy(image).float()
            
            # --- 关键修改 1: 掩码应为 Long 类型以进行 v2 变换 ---
            mask_tensor = torch.from_numpy(mask).long() 
            
            # --- 关键修改 2: 应用 v2 变换 (如果存在) ---
            # (这会替换掉原来对 sample 字典的 transform)
            if self.transform:
                image_tensor, mask_tensor = self.transform(image_tensor, mask_tensor)
            
            # --- 关键修改 3: 确保返回的掩码是 Float 类型 (如果损失函数需要) ---
            # 您的 DiceLoss 似乎期望 float 类型的标签，
            # 因为它在内部创建了 one-hot 编码 (label = torch.cat([bg, label1, label2], dim=1))
            mask_tensor = mask_tensor.float()

            # (删除旧的 "sample" 字典包装)
                
            return image_tensor, mask_tensor # 直接返回张量
        
        except Exception as e:
            logging.error(f"Error loading slice {idx} ({img_path}, slice {slice_idx}): {e}")
            return torch.zeros((3, 384, 384)), torch.zeros((1, 384, 384))