"""
ViT训练工具函数
包含数据增强、学习率调度等
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import random
import os


def set_seed(seed: int = 42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transforms(config):
    """
    获取训练数据增强
    
    Args:
        config: 配置对象
        
    Returns:
        transform: torchvision transforms
    """
    aug = config.train_augmentation
    
    transform_list = []
    
    # Random Resized Crop
    if aug.get('random_resized_crop', True):
        transform_list.append(
            transforms.RandomResizedCrop(
                config.image_size, 
                scale=aug.get('crop_scale', (0.8, 1.0))
            )
        )
    else:
        transform_list.append(transforms.Resize(config.image_size))
    
    # Horizontal Flip
    if aug.get('horizontal_flip', True):
        transform_list.append(transforms.RandomHorizontalFlip())
    
    # Color Jitter
    color_jitter = aug.get('color_jitter', None)
    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter.get('brightness', 0.4),
                contrast=color_jitter.get('contrast', 0.4),
                saturation=color_jitter.get('saturation', 0.4),
                hue=color_jitter.get('hue', 0.1)
            )
        )
    
    # Random Rotation
    rotation = aug.get('random_rotation', 0)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))
    
    # ToTensor and Normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(transform_list)


def get_val_transforms(config):
    """
    获取验证数据增强
    
    Args:
        config: 配置对象
        
    Returns:
        transform: torchvision transforms
    """
    aug = config.val_augmentation
    
    return transforms.Compose([
        transforms.Resize(aug.get('resize', 256)),
        transforms.CenterCrop(aug.get('center_crop', 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


class LRScheduler:
    """
    类比于ResNet训练方式
    Warmup + Cosine Annealing学习率调度器
    
    前warmup_epochs个epoch线性增长学习率
    之后使用cosine annealing衰减到min_lr
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
    
    def step(self, epoch: int):
        """更新学习率"""
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            lr = self.warmup_start_lr + \
                 (self.base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            # Cosine annealing阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        # 更新所有参数组的学习率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class SmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失
    
    将one-hot标签平滑化，防止过拟合
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 模型预测 (B, num_classes) logits
            target: 真实标签 (B,) class indices
            
        Returns:
            loss: 标量损失值
        """
        pred = pred.log_softmax(dim=-1)
        
        # 创建平滑后的标签
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # 计算KL散度
        loss = torch.sum(-true_dist * pred, dim=-1).mean()
        
        return loss


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    计算top-k准确率
    
    Args:
        output: 模型输出 (B, num_classes)
        target: 真实标签 (B,)
        topk: 要计算的top-k值元组：例如 (1, 5)
        
    Returns:
        res: top-k准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # 获取top-k预测
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        
        return res


def save_checkpoint(
    state: dict,
    save_dir: str,
    filename: str = 'checkpoint.pth'
):
    """
    保存模型checkpoint
    
    Args:
        state: 要保存的状态字典
        save_dir: 保存目录
        filename: 文件名
    """
    save_path = os.path.join(save_dir, "ckpt/")
    os.makedirs(save_path, exist_ok=True)

    filepath = os.path.join(save_path, filename)
    torch.save(state, filepath)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer=None
):
    """
    加载模型checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
        model: 要加载权重的模型
        optimizer: 可选的优化器
        
    Returns:
        start_epoch: 起始epoch
        best_acc: 最佳准确率
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"✓ Loaded checkpoint from epoch {start_epoch}, best_acc={best_acc:.2f}%")
    
    return start_epoch, best_acc


def get_parameter_groups(model, config):
    """
    获取不同学习率的参数组
    
    Args:
        model: ViT模型
        config: 配置对象
        
    Returns:
        param_groups: 参数组列表
    """
    # 分类头参数
    head_params = list(model.head.parameters())
    
    # Backbone参数（除了分类头）
    backbone_params = [
        p for n, p in model.named_parameters() 
        if 'head' not in n and p.requires_grad
    ]
    
    param_groups = [
        {'params': backbone_params, 'lr': config.stage2_lr_backbone, 'name': 'backbone'},
        {'params': head_params, 'lr': config.stage2_lr_head, 'name': 'head'}
    ]
    
    return param_groups


if __name__ == '__main__':
    # 测试工具函数
    print("Testing utility functions...")
    
    # 测试Label Smoothing
    criterion = SmoothingCrossEntropy(smoothing=0.1)
    pred = torch.randn(4, 10)
    target = torch.tensor([0, 2, 5, 9])
    loss = criterion(pred, target)
    print(f"Label Smoothing Loss: {loss.item():.4f}")
    
    # 测试accuracy
    output = torch.randn(32, 200)
    target = torch.randint(0, 200, (32,))
    top1, top5 = accuracy(output, target, topk=(1, 5))
    print(f"Top-1 Acc: {top1:.2f}%, Top-5 Acc: {top5:.2f}%")
    
    print("✓ All utility tests passed!")
