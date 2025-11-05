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
    crop_size = aug.get('random_resized_crop', 224)
    if crop_size:
        transform_list.append(transforms.RandomResizedCrop(crop_size))

    # RandAugment（如有配置）
    randaug = aug.get('rand_augment', None)
    if randaug is not None:
        n = randaug.get('n', 2)
        m = randaug.get('m', 9)
        transform_list.append(transforms.RandAugment(num_ops=n, magnitude=m))

    # Horizontal Flip
    if aug.get('horizontal_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip())

    # Color Jitter
    color_jitter = aug.get('color_jitter', None)
    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=color_jitter.get('brightness', 0.2),
                contrast=color_jitter.get('contrast', 0.2),
                saturation=color_jitter.get('saturation', 0.2),
                hue=color_jitter.get('hue', 0)
            )
        )

    # Random Rotation
    rotation = aug.get('random_rotation', 0)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))

    # ToTensor
    transform_list.append(transforms.ToTensor())

    # Normalize
    normalize = aug.get('normalize', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    transform_list.append(
        transforms.Normalize(
            mean=normalize.get('mean', [0.485, 0.456, 0.406]),
            std=normalize.get('std', [0.229, 0.224, 0.225])
        )
    )

    # Random Erasing (must be after ToTensor and Normalize)
    random_erasing = aug.get('random_erasing', None)
    if random_erasing:
        transform_list.append(
            transforms.RandomErasing(
                p=random_erasing.get('p', 0.5),
                scale=random_erasing.get('scale', (0.02, 0.2)),
                ratio=random_erasing.get('ratio', (0.3, 3.3)),
                value=random_erasing.get('value', 0)
            )
        )

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
    
    transform_list = [
        transforms.Resize(aug.get('resize', 256)),
        transforms.CenterCrop(aug.get('center_crop', 224)),
        transforms.ToTensor(),
    ]
    
    # Normalize
    normalize = aug.get('normalize', {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
    transform_list.append(
        transforms.Normalize(
            mean=normalize.get('mean', [0.485, 0.456, 0.406]),
            std=normalize.get('std', [0.229, 0.224, 0.225])
        )
    )
    
    return transforms.Compose(transform_list)


class LRScheduler:
    """
    连续训练的学习率调度器
    Warmup + Cosine Annealing学习率调度器
    
    支持多参数组，每个参数组可以有不同的学习率缩放因子
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
        
        # 保存每个参数组的学习率缩放因子
        self.lr_scales = []
        for param_group in optimizer.param_groups:
            scale = param_group.get('lr', base_lr) / base_lr
            self.lr_scales.append(scale)
    
    def step(self, epoch: int):
        """更新学习率"""
        if epoch < self.warmup_epochs:
            # Warmup阶段：线性增长
            base_lr_current = self.warmup_start_lr + \
                 (self.base_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            # Cosine annealing阶段
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            base_lr_current = self.min_lr + (self.base_lr - self.min_lr) * \
                 0.5 * (1 + np.cos(np.pi * progress))
        
        # 更新所有参数组的学习率（应用各自的缩放因子）
        lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i < len(self.lr_scales):
                lr = base_lr_current * self.lr_scales[i]
            else:
                lr = base_lr_current
            param_group['lr'] = lr
            lrs.append(lr)
        
        return lrs if len(lrs) > 1 else lrs[0]
    
    def get_last_lr(self):
        """获取当前学习率"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


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
    model: nn.Module
):
    """
    加载模型checkpoint
    
    Args:
        checkpoint_path: checkpoint文件路径
        model: 要加载权重的模型
        optimizer: 可选的优化器
        scheduler: 可选的学习率调度器
        
    Returns:
        checkpoint_info: 字典，包含 epoch, stage, best_acc, scheduler_state 等信息
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 不在此处直接加载优化器/调度器，交由调用方在对齐参数组后再恢复
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
    scheduler_state = checkpoint.get('scheduler_state', None)
    
    # 提取checkpoint信息
    start_epoch = checkpoint.get('epoch', 0)
    current_stage = checkpoint.get('stage', 1)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    print(f"✓ Loaded checkpoint from epoch {start_epoch}, stage {current_stage}, best_acc={best_acc:.2f}%")
    
    return {
        'epoch': start_epoch,
        'stage': current_stage,
        'best_acc': best_acc,
        'scheduler_state': scheduler_state,
        'optimizer_state_dict': optimizer_state_dict
    }


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


def get_unfrozen_backbone_params(model):
    params = [
        p for n, p in model.named_parameters() 
        if 'head' not in n and p.requires_grad
    ]
    return params


def update_optimizer_param_groups(optimizer, model, config, stage: int = 2):
    """
    动态更新优化器的参数组
    多阶段训练使用
    
    Args:
        optimizer: 优化器
        model: 模型
        stage: 当前阶段 (1, 2, 3)
        config: 配置对象
    """
    if stage == 2:
        # Stage 2: 添加解冻的backbone参数
        unfrozen_params = get_unfrozen_backbone_params(model)
        if unfrozen_params:
            optimizer.add_param_group({
                'params': unfrozen_params,
                'lr': config.stage2_lr_backbone,
                'name': 'backbone',
                'weight_decay': config.weight_decay
            })
        # 更新分类头学习率
        if len(optimizer.param_groups) > 0:
            optimizer.param_groups[0]['lr'] = config.stage2_lr_head
            
    elif stage == 3:
        # Stage 3: 添加“新”解冻参数，并统一学习率
        # 1) 已在优化器中的参数集合（用id判断）
        existing_param_ids = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                existing_param_ids.add(id(p))
        
        # 2) 当前可训练的backbone参数（使用原有工具函数）
        unfrozen_params = get_unfrozen_backbone_params(model)
        
        # 3) 过滤出本阶段“新”解冻但尚未被优化器管理的参数
        new_params = [p for p in unfrozen_params if id(p) not in existing_param_ids]
        
        # 4) 如有新参数，则添加一个新的参数组（避免与Stage 2重复）
        if len(new_params) > 0:
            optimizer.add_param_group({
                'params': new_params,
                'lr': config.stage3_lr,
                'name': 'backbone_stage3',
                'weight_decay': config.weight_decay
            })
        
        # 5) 统一所有参数组学习率为 stage3_lr（包括head与已有backbone组）
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.stage3_lr


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
