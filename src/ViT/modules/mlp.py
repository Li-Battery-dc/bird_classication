"""
MLP模块
"""

import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    """
    一层隐藏层的MLP
    
    结构:
        x -> Linear -> GELU -> Dropout -> Linear -> Dropout -> out
        
    维度变化:
        (B, N, embed_dim) -> (B, N, hidden_dim) -> (B, N, embed_dim)
    
    参数:
        in_features: 输入特征维度
        mlp_ratio: 隐藏层维度, hidden_dim = in_features * mlp_ratio
        out_features: 输出特征维度
        dropout: dropout比率 (默认0.0)
    """
    
    def __init__(
        self,
        in_features: int,
        mlp_ratio: float = 4.0,
        out_features: int = None,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # 如果未指定，hidden_features默认等于in_features
        # 在ViT中，通常设置为in_features * 4
        hidden_features = in_features * mlp_ratio
        
        # 如果未指定，out_features默认等于in_features
        out_features = out_features or in_features
        

        self.fc1 = nn.Linear(in_features, hidden_features)
        
        # GELU激活函数
        # ViT使用GELU而非ReLU，因为GELU在Transformer中表现更好
        # GELU(x) = x * Φ(x)，其中Φ是标准正态分布的累积分布函数
        self.act = GeLU()
        
        self.drop1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """        
        参数:
            x: 输入张量, 形状 (B, N, in_features)
               例如: (32, 197, 768)
        
        返回:
            output: 输出张量, 形状 (B, N, out_features)
                   例如: (32, 197, 768)
        
        """
        x = self.fc1(x)
        x = self.act(x)         # GELU激活
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        
        return x


class GeLU(nn.Module):
    """
    GELU(x) = x * Φ(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        GELU的近似计算
        """
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))