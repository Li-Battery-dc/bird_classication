"""
Transformer Encoder Block
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .mlp import MLP

from timm.models.layers import DropPath

class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block
    
    结构 (Pre-Norm):
        x -> LayerNorm -> MultiHeadAttention -> Add (residual) -> 
        x -> LayerNorm -> MLP -> Add (residual) -> out
    
    参数:
        embed_dim: 嵌入维度 (默认768)
        num_heads: 注意力头数 (默认12)
        mlp_ratio: MLP隐藏层扩展比例 (默认4)
        dropout: dropout比率 (默认0.0)
        attn_dropout: attention dropout比率 (默认0.0)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        
        # 第一个LayerNorm (在Attention之前)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Multi-Head Self-Attention
        self.attn = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout
        )
        
        # Dropout (在残差连接前)
        self.drop1 = nn.Dropout(dropout)
        
        # 第二个LayerNorm (在MLP之前)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network (MLP)
        self.mlp = MLP(
            in_features=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Dropout (在残差连接前)
        self.drop2 = nn.Dropout(dropout)

        # Stochastic Depth (DropPath)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        x -> LayerNorm -> MultiHeadAttention -> Add (residual) -> 
        x -> LayerNorm -> MLP -> Add (residual) -> out
        
        参数:
            x: 输入张量, 形状 (B, N, embed_dim)
            return_attention: 是否返回attention权重
        返回:
            output: 输出张量, 形状 (B, N, embed_dim)
            attention: (可选) 注意力权重
        """
        # 层1：
        shortcut = x
        x = self.norm1(x)
        if return_attention:
            x, attn = self.attn(x, return_attention=True)
        else:
            x = self.attn(x)
            attn = None
        x = self.drop1(x)
        # Residual connection
        x = self.drop_path(x) + shortcut
        
        # 层2：
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop2(x)
        x = self.drop_path(x) + shortcut
        
        if return_attention:
            return x, attn
        return x