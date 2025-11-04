"""
Vision Transformer (ViT) 模块
用于CUB-200鸟类分类任务
"""

from .modules.ViT_model import VisionTransformer, create_vit_base_patch16
from .config import ViTConfig
from .train import train_main

__version__ = '1.0.0'

__all__ = [
    'VisionTransformer',
    'create_vit_base_patch16',
    'ViTConfig',
    'train_main',
]
