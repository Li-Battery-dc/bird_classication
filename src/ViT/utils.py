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
    根据epoch区间调整学习率
    Warmup + Cosine Annealing学习率调度器
    Layer-wise Cosine LR Scheduler with Warmup and Layer Decay
    自动按照模型层数调整每层的学习率
    
    参数：
        optimizer: torch.optim.Optimizer
        base_lr: float, 基础学习率(head层)
        min_lr: float, 
        warmup_epochs: int, warmup 轮数
        start_epoch: int, 训练起始epoch
        end_epoch: int, 训练结束epoch
        warmup_start_lr: float, warmup起始学习率 (default=1e-6)
        layer_decay: float, 每层学习率衰减比例 (default=0.75)
        num_layers: int, 模型层数 (default=12)
        last_epoch: int, 上次训练的最后一个epoch (default=-1)
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        start_epoch: int,
        end_epoch: int,
        base_lr: float,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-6,
        layer_decay: float = 0.75,
        num_layers: int = 12
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.total_epochs = end_epoch - start_epoch + 1

        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.layer_decay = layer_decay
        self.num_layers = num_layers
        
        # 不用了，因为in optimizer param_groups里已经有scale了
        # # 逐层scale因子
        # self.layer_scales = [
        #     self.layer_decay ** (self.num_layers - i)
        #     for i in range(self.num_layers)
        # ]
        # self.layer_scales.append(1.0) # head

    def get_lr(self, epoch):
        # 当前 epoch 在区间中的偏移
        current_epoch_bias = epoch - self.start_epoch
        if current_epoch_bias < self.warmup_epochs:
            # 线性 warmup
            current_base_lr = self.warmup_start_lr + \
                (self.base_lr - self.warmup_start_lr) * (current_epoch_bias / self.warmup_epochs)
        else:
            # Cosine decay
            progress = (current_epoch_bias - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
            current_base_lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

        # 按每个参数组自身的 lr_scale 字段缩放，避免依赖 param_groups 顺序
        lrs = [current_base_lr * pg.get('lr_scale', 1.0) for pg in self.optimizer.param_groups]
        return lrs
    
    def step(self, epoch=None):
        """更新学习率"""
        new_lrs = self.get_lr(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, new_lrs):
            param_group['lr'] = lr

        return new_lrs
    
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
        支持soft label float 传入
        Args:
            pred: 模型预测 (B, num_classes) logits
            target: 真实标签 (B,) class indices
            
        Returns:
            loss: 标量损失值
        """
        pred = pred.log_softmax(dim=-1)
        
        # 如果target已经是soft label, 不需要标签平滑
        if target.ndim == 2:
            true_dist = target
        else:
            # 否则进行标签平滑
            with torch.no_grad():
                true_dist = torch.zeros_like(pred)
                true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
                true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        loss = torch.sum(-true_dist * pred, dim=-1).mean()
        return loss


def get_parameter_groups(model, base_lr, layer_decay=0.75, weight_decay=0.1):
    """
    按 Transformer Block 层级构建 param groups，实现 Layer-wise LR Decay
    并对 bias 和 LayerNorm 参数不使用 weight decay
    """

    parameter_group_names = {}
    parameter_group_vars = {}

    num_layers = model.depth
    # +1 给 head
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers)] + [1.0]

    def get_layer_id_for_vit(name):
        if name.startswith("patch_embed"):
            return 0  # 输入层
        elif name.startswith("blocks"):
            layer_id = int(name.split('.')[1])
            return layer_id + 1
        elif name.startswith("norm"):
            return num_layers  # norm在最后
        else:
            return num_layers + 1  # head层

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id_for_vit(name)
       

        # === 按参数类型划分 (decay / no_decay) ===
        if name.endswith("bias") or "norm" in name or "bn" in name:
            decay_type = "no_decay"
        else:
            decay_type = "decay"

        group_name = f"layer_{layer_id}_{decay_type}"

        if group_name not in parameter_group_names:
            scale = layer_scales[layer_id] if layer_id < len(layer_scales) else 1.0
            parameter_group_names[group_name] = {
                "params": [],
                "lr_scale": scale,
                "weight_decay": 0.0 if decay_type == "no_decay" else weight_decay,
            }
            parameter_group_vars[group_name] = []

        parameter_group_names[group_name]["params"].append(param)
        parameter_group_vars[group_name].append(param)

    # for k, v in parameter_group_names.items():
    #     print(f"Layer group: {k}, lr_scale={v['lr_scale']}, param_count={len(v['params'])}")

    # 构建 optimizer param groups
    param_groups = []
    for name, group in parameter_group_names.items():
        scale = group["lr_scale"]
        param_groups.append({
            "params": group["params"], 
            "lr_scale": scale, 
            "lr": base_lr * scale,
            "weight_decay": group["weight_decay"],
            "name": name
        })

    return param_groups


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """
    计算top-k准确率
    
    Args:
        output: 模型输出 (B, num_classes)
        target: 真实标签 (B,), soft label时为 (B, num_classes)
        topk: 要计算的top-k值元组：例如 (1, 5)
        
    Returns:
        res: top-k准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        # 获取top-k预测
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)

        if target.ndim == 2:  # soft label 情况, 训练时使用
            # pred shape: (B, maxk), target shape: (B, num_classes)
            # 找到每个样本soft label中概率最大的类别（作为真实标签）
            _, target_class = target.max(dim=1)  # (B,)
            # 然后按正常方式计算准确率
            correct = pred.eq(target_class.view(-1, 1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:, :k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return res
        else:
            # 普通 hard label 情况， 验证时使用
            correct = pred.eq(target.view(-1, 1).expand_as(pred))  # (B, maxk)
            res = []
            for k in topk:
                correct_k = correct[:, :k].reshape(-1).float().sum(0, keepdim=True)
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
