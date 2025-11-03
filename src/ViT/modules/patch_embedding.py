"""
Patch Embedding
将输入图像分割成patches并进行线性投影
添加位置编码和CLS token
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    将图像转换为patch embeddings
    
    工作流程:
    1. 输入图像 (B, C, H, W) - 例如 (32, 3, 224, 224)
    2. 分块 (B, embed_dim, H/P, W/P) - 例如 (32, 768, 14, 14)
    3. 展平patches (B, embed_dim, num_patches) - 例如 (32, 768, 196)
    4. 转置 (B, num_patches, embed_dim) - 例如 (32, 196, 768)
    5. CLS token (B, num_patches+1, embed_dim) - 例如 (32, 197, 768)
    6. 位置编码 (B, num_patches+1, embed_dim) - 例如 (32, 197, 768)
    
    参数:
        img_size: 输入图像大小 (默认224)
        patch_size: patch大小 (默认16)
        in_channels: 输入通道数 (默认3, RGB图像)
        embed_dim: 嵌入维度 (默认768, ViT-B的标准配置)
        dropout: dropout比率 (默认0.0)
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # 计算patch数量
        # 对于224x224的图像和16x16的patch: (224/16) * (224/16) = 14 * 14 = 196
        assert img_size % patch_size == 0, \
            f"图像大小 {img_size} 必须能被patch大小 {patch_size} 整除"
        
        self.num_patches = (img_size // patch_size) ** 2  # 196 for 224x224 with 16x16 patches
        self.grid_size = img_size // patch_size  # 14 for 224x224 with 16x16 patches
        
        # 使用卷积层实现patch embedding
        # 这等价于将图像分块后展平并进行线性投影
        # 卷积核大小 = patch_size, 步长 = patch_size (非重叠)
        # 输入: (B, 3, 224, 224) -> 输出: (B, 768, 14, 14)
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # CLS token
        # 形状: (1, 1, embed_dim)， 初始化为0
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        # 形状: (1, num_patches + 1, embed_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Dropout层用于正则化
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化权重
        - CLS token和位置编码使用截断正态分布初始化
        - 投影层使用默认的kaiming初始化
        """
        # 初始化CLS token (标准差0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化位置编码 (标准差0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，完成patch embedding
        
        参数:
            x: 输入图像张量, 形状 (B, C, H, W)
               例如: (32, 3, 224, 224)
        
        返回:
            patch embeddings: 形状 (B, num_patches + 1, embed_dim)
                             例如: (32, 197, 768)
        
        """
        B, C, H, W = x.shape
        
        # 验证输入尺寸
        assert H == self.img_size and W == self.img_size, \
            f"输入图像尺寸 ({H}, {W}) 与期望的 ({self.img_size}, {self.img_size}) 不匹配"
        
        # patch embedding
        # (B, C, H, W) -> (B, embed_dim, H/patch_size, W/patch_size)
        x = self.proj(x)
        
        # 展平patches
        # (B, embed_dim, H', W') -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        # 添加CLS token
        # cls_token: (1, 1, embed_dim) -> (B, 1, embed_dim)
        # 然后拼接: (B, num_patches, embed_dim) -> (B, num_patches+1, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # 添加位置编码
        # 直接相加: (B, num_patches+1, embed_dim) + (1, num_patches+1, embed_dim)
        x = x + self.pos_embedding

        # dropout
        x = self.dropout(x)
        
        return x
    
    def get_num_patches(self) -> int:
        """返回patch数量（不包括CLS token）"""
        return self.num_patches
    
    def get_grid_size(self) -> int:
        """返回patch grid的大小（每边的patch数）"""
        return self.grid_size
