import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNetwork(nn.Module):
    def __init__(self, num_classes=200):
        super(CNNNetwork, self).__init__()
        self.cov = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.resblocks = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 128, stride=2), # 使用 stride=2 进行下采样
            ResBlock(128, 128, use_drop=True, drop_prob=0.1, block_size=7),
            ResBlock(128, 256, stride=2, use_drop=True, drop_prob=0.1, block_size=7),
            ResBlock(256, 256, use_drop=True, drop_prob=0.1, block_size=7),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cov(x)
        x = self.resblocks(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_drop=False, drop_prob=0.1, block_size=7):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1), # 这个stride用来下采样
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.use_drop = use_drop
        if self.use_drop:
            self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        out = self.block(x)
        if self.use_drop:
            out = self.dropblock(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out

class DropBlock2D(nn.Module):
    '''
    参考 https://arxiv.org/abs/1810.12890
    使用 DropBlock 来替代 Dropout 以提高其泛化性
    用block_size x block_size 的正方形区域mask来Drop掉特征图上的连续区域
    '''

    def __init__(self, drop_prob=0.1, block_size=7):
        super().__init__()
        self.drop_prob = drop_prob  
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        # 根据论文定义计算 gamma
        h,w = x.shape[2], x.shape[3]
        gamma = (self.drop_prob * h * w) / ( # numel为元素总数， batch 维度
                  (h - self.block_size + 1) * (w - self.block_size + 1) *
                  (self.block_size ** 2)
                  )

        # 生成 Bernoulli mask：每个像素以 gamma 概率为 1
        mask = torch.bernoulli(torch.full_like(x, gamma))

        # 用 MaxPool2d 把 1 “膨胀”成 block_size×block_size 的方块
        block_mask = F.max_pool2d(
            mask, kernel_size=self.block_size,
            stride=1, padding=self.block_size // 2)

        # block 内像素丢弃
        block_mask = 1 - torch.clamp(block_mask, 0, 1)

        # 归一化：保持期望数值不变
        normalizer = block_mask.numel() / (block_mask.sum() + 1e-7)
        return x * block_mask * normalizer