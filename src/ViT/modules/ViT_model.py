"""
Vision Transformer架构
"""

import torch
import torch.nn as nn
from typing import Optional
from .patch_embedding import PatchEmbedding
from .transformer_block import TransformerBlock

import timm
import os


class VisionTransformer(nn.Module):
    """
    Vision Transformer模型
    
    架构流程:
    Input Image (B, 3, 224, 224)
        ↓
    Patch Embedding (B, 197, 768)  # 196 patches + 1 CLS token
        ↓
    Transformer Encoder  * depth
        ├─ Multi-Head Attention
        ├─ Layer Norm + Residual
        ├─ MLP
        └─ Layer Norm + Residual
        ↓
    Extract CLS Token (B, 768)
        ↓
    Layer Norm
        ↓
    Classification Head (B, num_classes)
    
    参数:
        img_size: 输入图像大小
        patch_size: Patch大小
        in_channels: 输入通道数
        num_classes: 分类类别数
        embed_dim: Embedding维度
        depth: Transformer Block层数
        num_heads: 注意力头数
        mlp_ratio: MLP隐藏层维度倍数
        dropout: Dropout比例
        attention_dropout: Attention dropout比例
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 200,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # 1. Patch Embedding层
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        num_patches = self.patch_embed.num_patches
        
        # 2. 堆叠Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attention_dropout
            )
            for _ in range(depth)
        ])
        
        # 3. 最终的Layer Norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. 分类头
        self.head = nn.Linear(embed_dim, num_classes)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化patch embedding的投影层
        nn.init.xavier_uniform_(self.patch_embed.proj.weight)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        
        # 初始化CLS token和位置编码
        nn.init.normal_(self.patch_embed.cls_token, std=0.02)
        nn.init.normal_(self.patch_embed.pos_embedding, std=0.02)
        
        # 初始化分类头
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            logits: 分类logits (B, num_classes)
        """
        # 1. Patch Embedding
        x = self.patch_embed(x)

        # 2. Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # 3. Layer Norm
        x = self.norm(x)
        
        # 4. 提取CLS token (第一个token)
        cls_token = x[:, 0]  # (B, embed_dim)
        
        # 5. 分类头
        logits = self.head(cls_token)  # (B, num_classes)
        
        return logits
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征（不经过分类头）
        用于特征可视化
        
        Args:
            x: 输入图像 (B, C, H, W)
            
        Returns:
            features: CLS token特征 (B, embed_dim)
        """
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token = x[:, 0]
        
        return cls_token
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1):
        """
        获取指定层的attention maps
        用于可视化注意力机制
        
        Args:
            x: 输入图像 (B, C, H, W)
            layer_idx: Transformer Block索引 (-1表示最后一层)
            
        Returns:
            attention_maps: 注意力图 (B, num_heads, num_patches+1, num_patches+1)
        """
        x = self.patch_embed(x)
        
        # 遍历到指定层
        target_idx = layer_idx if layer_idx >= 0 else self.depth + layer_idx
        
        for idx, block in enumerate(self.blocks):
            if idx == target_idx:
                # 获取该层的attention
                _, attention = block.attn(
                    block.norm1(x),
                    return_attention=True
                )
                return attention
            x = block(x)
        
        return None
    
    def freeze_backbone(self):
        """冻结Backbone（除了分类头）"""
        # 冻结patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # 冻结所有transformer blocks
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # 冻结norm层
        for param in self.norm.parameters():
            param.requires_grad = False
        
        # 分类头保持可训练
        for param in self.head.parameters():
            param.requires_grad = True
        
        print("✓ Backbone frozen, only classification head is trainable")
    
    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        解冻最后N个Transformer Blocks
        
        Args:
            n: 解冻的block数量
        """
        # 先冻结所有
        self.freeze_backbone()
        
        # 解冻最后n个blocks
        for block in self.blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        
        # 解冻最终的norm层
        for param in self.norm.parameters():
            param.requires_grad = True
        
        print(f"✓ Last {n} Transformer Blocks unfrozen")
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
        
        print("✓ All parameters unfrozen")
    
    def get_trainable_params(self):
        """获取可训练参数信息"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'trainable': trainable,
            'total': total,
            'frozen': total - trainable,
            'trainable_ratio': trainable / total
        }
    
    def print_trainable_params(self):
        """打印可训练参数信息"""
        info = self.get_trainable_params()
        print(f"Trainable Parameters: {info['trainable']:,} / {info['total']:,} "
              f"({info['trainable_ratio']*100:.2f}%)")
        print(f"Frozen Parameters: {info['frozen']:,}")


def _load_pretrained_weights(
    model: nn.Module,
    model_name: str = 'vit_base_patch16_224_in21k',
):
    """
    从timm加载预训练权重并映射到我们的模型
    
    Args:
        model: 我们的ViT模型
        model_name: timm模型名称
            - 'vit_base_patch16_224_in21k': ImageNet-21k预训练 (推荐)
            - 'vit_base_patch16_224': ImageNet-1k预训练
            - 'vit_base_patch16_384': 384x384输入
    
    Returns:
        成功加载的参数数量
    """
    try:
        pretrained_model = timm.create_model(model_name, pretrained=True)
        pretrained_dict = pretrained_model.state_dict()
    except Exception as e:
        print(f"⚠ Failed to download pretrained weights from timm: {e}")
        return 0
    
    # 键名映射：将timm的键名映射到我们实现的键名
    mapped_pretrained = {}
    for k, v in pretrained_dict.items():
        new_k = k
        # 将全局的cls_token/pos_embed映射到patch_embed内
        if new_k.startswith('cls_token'):
            new_k = new_k.replace('cls_token', 'patch_embed.cls_token')
        if new_k.startswith('pos_embed'):
            new_k = new_k.replace('pos_embed', 'patch_embed.pos_embedding')
        # 其他常见命名基本一致（patch_embed.proj, blocks, norm, head）
        mapped_pretrained[new_k] = v
    
    model_dict = model.state_dict()

    matched_dict = {}
    skipped_keys = []
    shape_mismatch = []
    
    # 检查参数匹配
    for key, value in mapped_pretrained.items():
        if key in model_dict:
            if model_dict[key].shape == value.shape:
                matched_dict[key] = value
            else:
                shape_mismatch.append(
                    f"{key}: pretrained {value.shape} vs model {model_dict[key].shape}"
                )
        else:
            skipped_keys.append(key)
    
    # 更新权重
    model_dict.update(matched_dict)
    model.load_state_dict(model_dict, strict=False)
    
    # 打印加载统计
    total_params = len(model_dict)
    loaded_params = len(matched_dict)
    match_rate = loaded_params / total_params * 100
    
    print(f"\n Successfully loaded: {loaded_params}/{total_params} parameters ({match_rate:.1f}%)")
    
    if shape_mismatch:
        print(f"Warning: Shape mismatch ({len(shape_mismatch)} parameters):")
        for item in shape_mismatch[:3]:  # 只显示前3个
            print(f"  - {item}")
        if len(shape_mismatch) > 3:
            print(f"  ... and {len(shape_mismatch)-3} more")
    
    # 分类头通常会被跳过（num_classes不同）
    head_skipped = any('head' in k for k in skipped_keys)
    if head_skipped:
        print(f"✓ Classification head skipped (expected for num_classes={model.num_classes})")
    
    return loaded_params


def create_vit_base_patch16(config=None):
    """
    创建ViT-Base/16模型
    
    Args:
        config: ViTConfig配置对象
        
    Returns:
        model: ViT模型
    """
    # 优先使用config中的参数
    if config is not None:
        num_classes = config.num_classes
        pretrained = config.use_pretrained
        pretrained_model = config.pretrained_model
        img_size = config.image_size
        patch_size = config.patch_size
        embed_dim = config.embed_dim
        depth = config.depth
        num_heads = config.num_heads
        mlp_ratio = config.mlp_ratio
        dropout = config.dropout
        attention_dropout = config.attention_dropout
    else:
        # 使用默认值或传入的参数
        num_classes = 200
        pretrained = True
        pretrained_model = 'vit_base_patch16_224_in21k'
        img_size = 224
        patch_size = 16
        embed_dim = 768
        depth = 12
        num_heads = 12
        mlp_ratio = 4.0
        dropout = 0.1
        attention_dropout = 0.1
    
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout
    )
    
    if pretrained:
        loaded = _load_pretrained_weights(model, model_name=pretrained_model)
        if loaded == 0:
            print("⚠ Warning: No pretrained weights loaded, training from scratch")
    else:
        print("\nTraining from scratch (no pretrained weights)\n")
    
    return model