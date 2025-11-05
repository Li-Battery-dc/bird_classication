import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from timm.data.mixup import Mixup

import os
import json
import copy
import sys
import time
import datetime
from tqdm import tqdm

from dataloader.Dataloader import DataLoader
from .modules.ViT_model import create_vit_base_patch16
from .config import ViTConfig
from .utils import (
    set_seed, get_train_transforms, get_val_transforms,
    LRScheduler, SmoothingCrossEntropy, get_parameter_groups,
    accuracy, save_checkpoint, load_checkpoint
)

def train_one_epoch(
    model, 
    data_loader, 
    criterion, 
    optimizer, 
    device, 
    epoch, 
    config,
    batch_size=64, # 用于分阶段改变batch_size
    mixup_fn=None,
    is_scale=False
):
    """
    训练一个epoch
    
    Args:
        model: ViT模型
        data_loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        config: 配置对象
        
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    model.train()
    
    # 每个batch的统计
    loss_list = []
    # 两种准确率
    acc1_list = []
    acc5_list = []
    
    batch_iterator = data_loader.get_batch_iterator(
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # 保证每个batch都是完整的
    )
    
    # 计算总batch数（用于进度条）
    total_batches = len(data_loader) // batch_size
    
    pbar = tqdm(batch_iterator, total=total_batches, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # 转换为tensor并移到设备
        if isinstance(images, torch.Tensor):
            images = images.to(device)
        else:
            images = torch.tensor(images, dtype=torch.float32).to(device)
        
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        # 应用MixUp/CutMix, label会变成soft label
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        if is_scale:
            scaler = GradScaler(device=device)
            with autocast(device_type=device.type):
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, labels)

                scaler.scale(loss).backward()

                if config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
        else:
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
        
            # 梯度裁剪
            if config.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.max_grad_norm
                )
        
            optimizer.step()
        
        # 计算准确率
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        
        # 记录统计
        loss_list.append(loss.item())
        acc1_list.append(acc1)
        acc5_list.append(acc5)
        
        # 更新进度条 - 显示当前平均值
        if batch_idx % config.log_freq == 0:
            pbar.set_postfix({
                'loss': f'{sum(loss_list)/len(loss_list):.4f}',
                'acc1': f'{sum(acc1_list)/len(acc1_list):.2f}%',
                'acc5': f'{sum(acc5_list)/len(acc5_list):.2f}%'
            })
    
    # 计算整个epoch的平均值
    avg_loss = sum(loss_list) / len(loss_list)
    avg_acc1 = sum(acc1_list) / len(acc1_list)
    avg_acc5 = sum(acc5_list) / len(acc5_list)
    
    return avg_loss, avg_acc1, avg_acc5


def validate(model, data_loader, criterion, device, config):
    """
    验证模型
    
    Args:
        model: ViT模型
        data_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        config: 配置对象
        
    Returns:
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """
    model.eval()
    
    # 每个batch的统计信息
    loss_list = []
    acc1_list = []
    acc5_list = []
    
    batch_iterator = data_loader.get_batch_iterator(
        batch_size=config.batch_size_val,
        shuffle=False
    )

    total_batches = len(data_loader) // config.batch_size_val

    with torch.no_grad():
        for images, labels in tqdm(batch_iterator, total=total_batches, desc="Validating"):
            # 转换为tensor并移到设备
            if isinstance(images, torch.Tensor):
                images = images.to(device)
            else:
                images = torch.tensor(images, dtype=torch.float32).to(device)
            
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 计算准确率
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            # 记录统计
            loss_list.append(loss.item())
            acc1_list.append(acc1)
            acc5_list.append(acc5)
    
    # 计算平均值
    avg_loss = sum(loss_list) / len(loss_list)
    avg_acc1 = sum(acc1_list) / len(acc1_list)
    avg_acc5 = sum(acc5_list) / len(acc5_list)
    
    return avg_loss, avg_acc1, avg_acc5


def train_main():
    """
    连续训练主函数
    使用单一优化器和学习率调度器，在不同阶段动态调整参数
    """
    # 加载配置
    config = ViTConfig()
    
    # 设置随机种子
    set_seed(config.seed)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(config.result_dir, f'train_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving results to: {save_dir}")
    log_path = os.path.join(save_dir, 'train.log')
    
    def logger(msg: str):
        print(msg)
        with open(log_path, 'a') as f:
            f.write(str(msg) + "\n")
    
    # 数据增强
    mixup_fn = Mixup(
        mixup_alpha=config.mixup_params['mixup_alpha'],
        cutmix_alpha=config.mixup_params['cutmix_alpha'],
        prob=config.mixup_params['prob'],
        switch_prob=config.mixup_params['switch_prob'],
        label_smoothing=config.label_smoothing,
        num_classes=config.num_classes
    )
    train_transform = get_train_transforms(config)
    val_transform = get_val_transforms(config)

    logger("\n" + "=" * 60)
    logger("Loading Data")
    logger("=" * 60)

    # 加载数据
    train_loader = DataLoader(
        data_root=config.data_root,
        split='train',
        mode='image',
        transform=train_transform
    )
    
    val_loader = DataLoader(
        data_root=config.data_root,
        split='val',
        mode='image',
        transform=val_transform
    )
    
    logger(f"Train samples: {len(train_loader)}")
    logger(f"Val samples: {len(val_loader)}")
    
    # 创建模型
    logger("\n" + "=" * 60)
    logger("Creating Model")
    logger("=" * 60)
    
    model = create_vit_base_patch16(config=config)
    model = model.to(device)
    
    # 初始冻结backbone
    model.freeze_backbone()
    model.print_trainable_params()
    
    # 计算总训练轮数
    total_epochs = config.stage1_epochs + config.stage2_epochs + config.stage3_epochs
    stage1_end = config.stage1_epochs
    stage2_end = stage1_end + config.stage2_epochs
    
    logger("\n" + "=" * 80)
    logger("Training Configuration (Continuous Mode)")
    logger("=" * 80)
    logger(f"Total Epochs: {total_epochs}")
    logger(f"  - Stage 1 (Head only): Epochs 1-{stage1_end}")
    if config.stage2_epochs > 0:
        logger(f"  - Stage 2 (Unfreeze {config.stage2_unfreeze_layers} blocks): Epochs {stage1_end+1}-{stage2_end}")
    if config.stage3_epochs > 0:
        logger(f"  - Stage 3 (Unfreeze {config.stage3_unfreeze_layers} blocks): Epochs {stage2_end+1}-{total_epochs}")
    logger("=" * 80)
    
    # 创建损失函数
    criterion = SmoothingCrossEntropy(smoothing=config.label_smoothing)
    
    # 创建优化器
    # 初始化只包含分类头, 没用get_parameter_groups加scale
    optimizer = optim.AdamW(
        [{'params': model.head.parameters(), 'lr': config.stage1_base_lr, 'name': 'head'}],
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    scheduler = LRScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=total_epochs,
        base_lr=config.stage1_base_lr,
        min_lr=config.min_lr,
        warmup_start_lr=config.warmup_start_lr,
        layer_decay=config.layer_decay,
        num_layers=model.depth
    )
    
    # 训练状态
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    current_stage = 1
    start_epoch = 1
    
    # === 从checkpoint恢复训练 ===
    if config.resume_from_checkpoint is not None and config.resume_from_checkpoint != "":
        logger("\n" + "=" * 80)
        logger("Resuming from Checkpoint")
        logger("=" * 80)
        try:
            checkpoint_info = load_checkpoint(
                config.resume_from_checkpoint,
                model
            )

            start_epoch = checkpoint_info['epoch'] + 1  # 从下一个epoch开始
            current_stage = checkpoint_info['stage']
            best_acc = checkpoint_info['best_acc']
            best_state = copy.deepcopy(model.state_dict())

            # 先根据恢复的stage调整模型状态与可训练层
            if current_stage == 1:
                model.freeze_backbone()
                batch_size = config.stage1_batch_size
                is_scale = False
            elif current_stage == 2:
                model.unfreeze_last_n_blocks(config.stage2_unfreeze_layers)
                batch_size = config.stage2_batch_size
                is_scale = True
                param_groups = get_parameter_groups(model, base_lr=config.stage2_base_lr, layer_decay=config.layer_decay)
            elif current_stage == 3:
                model.unfreeze_last_n_blocks(config.stage3_unfreeze_layers)
                batch_size = config.stage3_batch_size
                is_scale = True
                param_groups = get_parameter_groups(model, base_lr=config.stage3_base_lr, layer_decay=config.layer_decay)

            optimizer.param_groups = param_groups
            # 恢复优化器与调度器状态（在参数组对齐之后）
            opt_state = checkpoint_info.get('optimizer_state_dict', None)
            if opt_state is not None:
                try:
                    optimizer.load_state_dict(opt_state)
                    logger("✓ Optimizer state loaded after aligning param groups")
                except ValueError as ve:
                    logger(f"⚠ Optimizer state restore skipped (param groups mismatch): {ve}")
            # 恢复scheduler lr_scales（如存在）
            sched_state = checkpoint_info.get('scheduler_state', None)
            if sched_state and 'lr_scales' in sched_state:
                scheduler.lr_scales = sched_state['lr_scales']
                logger(f"✓ Scheduler lr_scales restored: {scheduler.lr_scales}")

            logger(f"✓ Resuming from epoch {start_epoch}, stage {current_stage}")
            logger(f"✓ Best accuracy so far: {best_acc:.2f}%")
            logger("=" * 80)
            
        except Exception as e:
            logger(f"⚠ Failed to load checkpoint: {e}")
            logger("⚠ Starting training from scratch")
            start_epoch = 1
            current_stage = 1
            best_acc = 0.0
    
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            batch_size = config.stage1_batch_size
            is_scale = False
            if epoch == 0 and mixup_fn is not None:
                logger(f"Using MixUp(alpha={config.mixup_params['mixup_alpha']}) + CutMix(alpha={config.mixup_params['cutmix_alpha']})")

            # === Stage切换逻辑 ===
            if epoch == stage1_end + 1 and config.stage2_epochs > 0 and current_stage < 2:
                logger(f"Entering Stage 2: Unfreezing last {config.stage2_unfreeze_layers} blocks")
                current_stage = 2
                batch_size = config.stage2_batch_size
                is_scale = True
                patience_counter = 0  # reset patience when entering new stage
                
                # 解冻后几层
                model.unfreeze_last_n_blocks(config.stage2_unfreeze_layers)
                model.print_trainable_params()

                # 更新优化器参数组
                param_group_2 = get_parameter_groups(
                    model,
                    base_lr=config.stage2_base_lr,
                    layer_decay=config.layer_decay
                )
                optimizer = optim.AdamW(
                    param_group_2,
                    weight_decay=config.weight_decay,
                    betas=config.betas
                )
                scheduler = LRScheduler(
                    optimizer,
                    warmup_epochs=config.warmup_epochs,
                    total_epochs=total_epochs,
                    base_lr=config.stage2_base_lr, # 使用阶段2的base_lr
                    min_lr=config.min_lr,
                    warmup_start_lr=config.warmup_start_lr,
                    layer_decay=config.layer_decay,
                    num_layers=model.depth
                )
                
                logger(f"✓ Optimizer updated with {len(optimizer.param_groups)} parameter groups")
                for i, pg in enumerate(optimizer.param_groups):
                    logger(f"  Group {i} ({pg.get('name', 'unknown')}): lr={pg['lr']:.6f}")
            
            if epoch == stage2_end + 1 and config.stage3_epochs > 0 and current_stage < 3:
                logger(f"Entering Stage 3: Unfreezing last {config.stage3_unfreeze_layers} blocks")
                current_stage = 3
                batch_size = config.stage3_batch_size
                is_scale = True
                patience_counter = 0  # reset patience when entering new stage
                # 只解冻最后N层
                model.unfreeze_last_n_blocks(config.stage3_unfreeze_layers)
                model.print_trainable_params()

                # 更新优化器参数组
                param_group_3 = get_parameter_groups(
                    model,
                    base_lr=config.stage3_base_lr,
                    layer_decay=config.layer_decay
                )
                optimizer = optim.AdamW(
                    param_group_3,
                    weight_decay=config.weight_decay,
                    betas=config.betas
                )
                scheduler = LRScheduler(
                    optimizer,
                    warmup_epochs=config.warmup_epochs,
                    total_epochs=total_epochs,
                    base_lr=config.stage3_base_lr, # 使用阶段3的base_lr
                    min_lr=config.min_lr,
                    warmup_start_lr=config.warmup_start_lr,
                    layer_decay=config.layer_decay,
                    num_layers=model.depth
                )
    
                logger(f"✓ Optimizer updated with {len(optimizer.param_groups)} parameter groups")
                for i, pg in enumerate(optimizer.param_groups):
                    logger(f"  Group {i} ({pg.get('name', 'unknown')}): lr={pg['lr']:.6f}")
            
            # === 更新学习率 ===
            current_lrs = scheduler.step(epoch - 1)
            
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, train_loader, criterion, optimizer, 
                device, epoch, config, batch_size, mixup_fn, is_scale
            )
            
            val_loss, val_acc1, val_acc5 = validate(
                model, val_loader, criterion, device, config
            )
            
            logger(f"\nEpoch {epoch}/{total_epochs} [Stage {current_stage}]")
            logger(f"Train - Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%")
            logger(f"Val   - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")
            
            # 打印各参数组的学习率
            if isinstance(current_lrs, list):
                for i, lr in enumerate(current_lrs):
                    group_name = optimizer.param_groups[i].get('name', f'group_{i}')
                    lr_scale = optimizer.param_groups[i].get('lr_scale', 1.0)
                    logger(f"LR ({group_name}): {lr:.6f} (scale={lr_scale:.3f})")
            else:
                logger(f"LR: {current_lrs:.6f}")
            
            # === 保存最佳模型 ===
            if val_acc1 > best_acc:
                best_acc = val_acc1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
            
            # === 早停检查 ===
            if config.early_stopping and patience_counter >= config.patience:
                logger(f"\n⚠ Early stopping triggered after {epoch} epochs (patience={config.patience})")
                break
            
            # === 定期保存checkpoint ===
            if config.save_freq > 0 and epoch % config.save_freq == 0:
                save_checkpoint(
                    state={
                        'epoch': epoch,
                        'stage': current_stage,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state': {
                            'lr_scales': scheduler.lr_scales,
                            'last_epoch': epoch
                        },
                        'best_acc': best_acc,
                    },
                    save_dir=save_dir,
                    filename=f'checkpoint_epoch_{epoch}.pth'
                )
    
    except KeyboardInterrupt:
        logger("\n⚠ Training interrupted by user (Ctrl+C)")
    
    finally:
        # === 保存最佳模型 ===
        if best_state is not None:
            save_checkpoint(
                state={
                    'epoch': epoch if 'epoch' in locals() else total_epochs,
                    'model_state_dict': best_state,
                    'best_acc': best_acc,
                },
                save_dir=save_dir,
                filename='best_model.pth'
            )
            logger(f"\n✓ Best model saved: {os.path.join(save_dir, 'best_model.pth')}")
            logger(f"✓ Best Acc@1: {best_acc:.2f}%")
        
        logger("\n" + "=" * 80)
        logger("Training Completed!")
        logger("=" * 80)
        logger(f"Results saved to: {save_dir}")


if __name__ == '__main__':
    train_main()
