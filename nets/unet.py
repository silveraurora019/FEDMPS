# 文件名: new_unet.py
# (已修复 inplace 错误)

from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# --- 关键修改: _block 函数中的 inplace=True 全部改为 False ---
def _block(in_channels, features, name, affine=True, track_running_stats=True, dropout_p=0.0):
    bn_func = nn.BatchNorm2d

    layers = [
        (
            name + "_conv1",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        ),
        (name + "_bn1", bn_func(num_features=features, affine=affine,
                                track_running_stats=track_running_stats)),
        
        # --- 修改 1: inplace=True 改为 inplace=False ---
        (name + "_relu1", nn.ReLU(inplace=False)), 
    ]
    
    if dropout_p > 0:
        # --- 修改 2: inplace=True 改为 inplace=False ---
        layers.append((name + "_drop1", nn.Dropout2d(p=dropout_p, inplace=False))) 

    layers.extend([
        (
            name + "_conv2",
            nn.Conv2d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
        ),
        (name + "_bn2", bn_func(num_features=features, affine=affine,
                                track_running_stats=track_running_stats)),
        
        # --- 修改 3: inplace=True 改为 inplace=False ---
        (name + "_relu2", nn.ReLU(inplace=False)), 
    ])
    
    if dropout_p > 0:
        # --- 修改 4: inplace=True 改为 inplace=False ---
        layers.append((name + "_drop2", nn.Dropout2d(p=dropout_p, inplace=False))) 

    return nn.Sequential(OrderedDict(layers))
# --- 修改结束 ---


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, aug_method=None, dropout_p=0.0):
        super(UNet, self).__init__()
        
        self.dropout_p = dropout_p 
        features = init_features
        
        # (自动传递 dropout_p 和新的 _block 设置)
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        bottleneck = self.bottleneck(enc4_)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)
        return dec1


class UNet_pro(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, init_features=32,
                 affine=True, track_running_stats=True, aug_method=None, dropout_p=0.0):
        super(UNet_pro, self).__init__()

        self.dropout_p = dropout_p 
        features = init_features
        
        # (自动传递 dropout_p 和新的 _block 设置)
        self.encoder1 = _block(in_channels, features, name="enc1", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = _block(features, features * 2, name="enc2", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = _block(features * 2, features * 4, name="enc3", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = _block(features * 4, features * 8, name="enc4", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = _block(features * 8, features * 16, name="bottleneck", affine=affine,
                                 track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = _block((features * 8) * 2, features * 8, name="dec4", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2,
        )
        self.decoder3 = _block((features * 4) * 2, features * 4, name="dec3", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = _block((features * 2) * 2, features * 2, name="dec2", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2,
        )
        self.decoder1 = _block(features * 2, features, name="dec1", affine=affine,
                               track_running_stats=track_running_stats, dropout_p=self.dropout_p)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1_ = self.pool1(enc1)
        enc2 = self.encoder2(enc1_)
        enc2_ = self.pool2(enc2)
        enc3 = self.encoder3(enc2_)
        enc3_ = self.pool3(enc3)
        enc4 = self.encoder4(enc3_)
        enc4_ = self.pool4(enc4)
        bottleneck = self.bottleneck(enc4_)
        # z 是瓶颈层的 avg_pool
        z = F.adaptive_avg_pool2d(bottleneck,2).view(bottleneck.shape[0],-1)
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # shadow 是解码器最后一层的特征
        shadow = dec1
        dec1 = self.conv(dec1)
        # 返回 (输出, z特征, shadow特征)
        return dec1, z, shadow