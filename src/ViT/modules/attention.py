"""
Multi-Head Self-Attention 模块
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    核心公式:
        Attention(Q, K, V) = softmax(QK^T / √d_k) V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
    pipeline:
    1. 输入 (B, N, D) 通过三个线性层得到 Q, K, V
    2. 将 Q, K, V 分成多个头 (B, num_heads, N, head_dim) 独立计算 attention
    3. 合并所有头的输出通过输出投影层
    
    参数:
        embed_dim
        num_heads
        dropout:
        qkv_bias: Q,K,V投影是否使用bias (默认True)
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.0,
        qkv_bias: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 确保embed_dim能被num_heads整除
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) 必须能被 num_heads ({num_heads}) 整除"
        
        # 每个头的维度
        # 例如: 768 / 12 = 64
        self.head_dim = embed_dim // num_heads
        
        # 缩放因子，防止softmax梯度过小
        self.scale = self.head_dim ** -0.5  # 1/√d_k
        
        # QKV投影层: 一次性生成Q, K, V, 然后会被拆分成Q, K, V三部分
        # 输入: (B, N, embed_dim)
        # 输出: (B, N, 3 * embed_dim) = (B, N, 2304) for embed_dim=768
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(dropout)
        
        # 输出投影层
        # 将多头的输出合并后投影回embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        前向传播，计算注意力分数: softmax(Q @ K^T / √d_k) V
        
        参数:
            x: 输入张量, 形状 (B, N, embed_dim)
               例如: (32, 197, 768)
            return_attention: 是否返回attention权重 (用于可视化)
        
        返回:
            output: 输出张量, 形状 (B, N, embed_dim)
            attention_weights: (可选) 注意力权重, 形状 (B, num_heads, N, N)
        """
        B, N, C = x.shape  # B=batch, N=seq_len, C=embed_dim
        
        # 生成Q, K, V
        # qkv形状: (B, N, 3*C) -> 例如 (32, 197, 2304)
        qkv = self.qkv(x)
        
        # 分离Q, K, V
        # (B, N, 3*C) -> (B, N, 3, num_heads, head_dim)
        # 例如: (32, 197, 2304) -> (32, 197, 3, 12, 64)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        # 调整维度顺序: (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        # 便于拆分
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # 拆分Q, K, V
        # 每个的形状: (B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 核心计算 Scaled Dot-Product Attention
        # Q @ K^T: (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N)
        #        -> (B, num_heads, N, N)
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # 缩放: 除以 √d_k
        attn = attn * self.scale
        
        # Softmax：
        # 对最后一个维度(key的维度)进行softmax，表示注意力权重分布
        attn = attn.softmax(dim=-1)
        
        # Dropout
        attn = self.attn_dropout(attn)
        
        # attn @ V: (B, num_heads, N, N) @ (B, num_heads, N, head_dim)
        #        -> (B, num_heads, N, head_dim)
        x = torch.matmul(attn, v)
        
        # 合并多个头
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim)
        x = x.transpose(1, 2)
        # (B, N, num_heads, head_dim) -> (B, N, embed_dim)
        x = x.reshape(B, N, C)
        
        # 步骤5: 输出投影
        # (B, N, embed_dim) -> (B, N, embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x
    
    def get_attention_map(self, x: torch.Tensor, head_idx: int = 0):
        """
        获取指定头的注意力图（用于可视化）
        
        参数:
            x: 输入张量, 形状 (B, N, embed_dim)
            head_idx: 要可视化的头的索引
        
        返回:
            attention_map: 注意力权重, 形状 (B, N, N)
        """
        with torch.no_grad():
            _, attn = self.forward(x, return_attention=True)
            # 返回指定头的注意力权重
            # attn形状: (B, num_heads, N, N)
            return attn[:, head_idx, :, :]