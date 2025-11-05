import torch
import torch.nn as nn
import torch.optim as optim

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
    LRScheduler, SmoothingCrossEntropy,
    accuracy, save_checkpoint, load_checkpoint, get_parameter_groups,
    update_optimizer_param_groups
)


def train_one_epoch(
    model, 
    data_loader, 
    criterion, 
    optimizer, 
    device, 
    epoch, 
    config
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
    
    # 使用批次迭代器
    batch_iterator = data_loader.get_batch_iterator(
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    # 计算总batch数（用于进度条）
    total_batches = len(data_loader) // config.batch_size
    
    pbar = tqdm(batch_iterator, total=total_batches, desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in enumerate(pbar):
        # 转换为tensor并移到设备
        if isinstance(images, torch.Tensor):
            images = images.to(device)
        else:
            images = torch.tensor(images, dtype=torch.float32).to(device)
        
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        
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
        batch_size=config.batch_size,
        shuffle=False
    )
    
    total_batches = len(data_loader) // config.batch_size
    
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
    # 初始化只包含分类头
    optimizer = optim.AdamW(
        [{'params': model.head.parameters(), 'lr': config.stage1_lr_head, 'name': 'head'}],
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    scheduler = LRScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=total_epochs,
        base_lr=config.stage1_lr_head,
        min_lr=config.min_lr,
        warmup_start_lr=config.warmup_start_lr
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
            elif current_stage == 2:
                model.unfreeze_last_n_blocks(config.stage2_unfreeze_layers)
                # 确保优化器参数组与阶段2匹配后再恢复优化器状态
                update_optimizer_param_groups(optimizer, model, config=config, stage=2)
            elif current_stage == 3:
                model.unfreeze_last_n_blocks(config.stage3_unfreeze_layers)
                update_optimizer_param_groups(optimizer, model, config=config, stage=3)

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
            # === Stage切换逻辑 ===
            if epoch == stage1_end + 1 and config.stage2_epochs > 0 and current_stage < 2:
                logger(f"Entering Stage 2: Unfreezing last {config.stage2_unfreeze_layers} blocks")
                current_stage = 2
                patience_counter = 0  # reset patience when entering new stage
                
                # 解冻后几层
                model.unfreeze_last_n_blocks(config.stage2_unfreeze_layers)
                model.print_trainable_params()
                
                # 动态添加backbone参数到优化器
                update_optimizer_param_groups(optimizer, model, config=config, stage=2)
                
                # 更新scheduler的学习率缩放因子
                scheduler.lr_scales = []
                for param_group in optimizer.param_groups:
                    scale = param_group['lr'] / config.stage1_lr_head
                    scheduler.lr_scales.append(scale)
                
                logger(f"✓ Optimizer updated with {len(optimizer.param_groups)} parameter groups")
                for i, pg in enumerate(optimizer.param_groups):
                    logger(f"  Group {i} ({pg.get('name', 'unknown')}): lr={pg['lr']:.6f}")
            
            if epoch == stage2_end + 1 and config.stage3_epochs > 0 and current_stage < 3:
                logger(f"Entering Stage 3: Unfreezing last {config.stage3_unfreeze_layers} blocks")
                current_stage = 3
                patience_counter = 0  # reset patience when entering new stage
                # 只解冻最后N层
                model.unfreeze_last_n_blocks(config.stage3_unfreeze_layers)
                model.print_trainable_params()
    
                # 动态添加backbone参数到优化器
                update_optimizer_param_groups(optimizer, model, config=config, stage=3)
                
                # 更新scheduler的学习率缩放因子
                scheduler.lr_scales = []
                scheduler.base_lr = config.stage3_lr
                for param_group in optimizer.param_groups:
                    scale = param_group['lr'] / scheduler.base_lr
                    scheduler.lr_scales.append(scale)
                
                logger(f"✓ Optimizer updated with {len(optimizer.param_groups)} parameter groups")
                for i, pg in enumerate(optimizer.param_groups):
                    logger(f"  Group {i} ({pg.get('name', 'unknown')}): lr={pg['lr']:.6f}")
            
            # === 更新学习率 ===
            current_lrs = scheduler.step(epoch - 1)
            
            train_loss, train_acc1, train_acc5 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, config
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
                    logger(f"LR ({group_name}): {lr:.6f}")
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
