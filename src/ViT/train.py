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
    accuracy, save_checkpoint, get_parameter_groups
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


def train_stage1(model, train_loader, val_loader, device, config, save_dir, logger=print):
    """
    阶段1: 只训练分类头
    
    Args:
        model: ViT模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        config: 配置对象
        save_dir: 保存目录
    """
    logger("\n" + "=" * 80)
    logger("Stage 1: Training Classification Head Only")
    logger("=" * 80)
    
    # 冻结backbone
    model.freeze_backbone()
    model.print_trainable_params()
    
    # 损失函数
    criterion = SmoothingCrossEntropy(smoothing=config.label_smoothing)
    
    # 优化器（只优化分类头）
    optimizer = optim.AdamW(
        model.head.parameters(),
        lr=config.stage1_lr_head,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    # 学习率调度器
    scheduler = LRScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.stage1_epochs,
        base_lr=config.stage1_lr_head,
        min_lr=config.min_lr
    )
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(1, config.stage1_epochs + 1):
        # 更新学习率
        current_lr = scheduler.step(epoch - 1)
        
        # 训练
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # 验证
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, config
        )
        
        # 记录日志
        logger(f"Epoch {epoch}/{config.stage1_epochs}")
        logger(f"Train - Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%")
        logger(f"Val   - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")
        logger(f"LR: {current_lr:.6f}")

        # 更新最佳
        if val_acc1 > best_acc:
            best_acc = val_acc1
            best_state = copy.deepcopy(model.state_dict())

        # 按频率保存普通checkpoint（可选）
        if config.save_freq and epoch % config.save_freq == 0:
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                },
                save_dir=save_dir,
                filename=f'stage1_epoch_{epoch}.pth'
            )
    
    # 在阶段结束时仅保存一次最佳模型
    if best_state is not None:
        save_checkpoint(
            state={
                'epoch': config.stage1_epochs,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            },
            save_dir=save_dir,
            filename='best_stage1.pth'
        )
    logger(f"\nStage 1 completed. Best Val Acc@1: {best_acc:.2f}%")
    return best_acc, best_state


def train_stage2(model, train_loader, val_loader, device, config, save_dir, logger=print):
    """
    阶段2: 微调后几层
    
    Args:
        model: ViT模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        config: 配置对象
        save_dir: 保存目录
    """
    logger("\n" + "=" * 80)
    logger(f"Stage 2: Fine-tuning Last {config.stage2_unfreeze_layers} Transformer Blocks")
    logger("=" * 80)
    
    # 解冻后几层
    model.unfreeze_last_n_blocks(config.stage2_unfreeze_layers)
    model.print_trainable_params()
    
    # 损失函数
    criterion = SmoothingCrossEntropy(smoothing=config.label_smoothing)
    
    # 优化器（不同学习率）
    param_groups = get_parameter_groups(model, config)
    optimizer = optim.AdamW(
        param_groups,
        weight_decay=config.weight_decay,
        betas=config.betas
    )
    
    # 学习率调度器
    scheduler = LRScheduler(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        total_epochs=config.stage2_epochs,
        base_lr=config.stage2_lr_backbone,  # 使用backbone的lr作为base
        min_lr=config.min_lr
    )
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, config.stage2_epochs + 1):
        # 更新学习率
        current_lr = scheduler.step(epoch - 1)
        
        # 训练
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config
        )
        
        # 验证
        val_loss, val_acc1, val_acc5 = validate(
            model, val_loader, criterion, device, config
        )
        
        # 打印统计
        logger(f"Epoch {epoch}/{config.stage2_epochs}")
        logger(f"Train - Loss: {train_loss:.4f}, Acc@1: {train_acc1:.2f}%, Acc@5: {train_acc5:.2f}%")
        logger(f"Val   - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")
        logger(f"LR: {current_lr:.6f}")
        
        # 早停检查
        if val_acc1 > best_acc:
            best_acc = val_acc1
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        if config.early_stopping and patience_counter >= config.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
        
        if config.save_freq and epoch % config.save_freq == 0:
            save_checkpoint(
                state={
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                },
                save_dir=save_dir,
                filename=f'stage2_epoch_{epoch}.pth'
            )
    
    # 阶段结束保存最佳
    if best_state is not None:
        save_checkpoint(
            state={
                'epoch': config.stage2_epochs,
                'model_state_dict': best_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': config
            },
            save_dir=save_dir,
            filename='best_stage2.pth'
        )
    logger(f"\nStage 2 completed. Best Val Acc@1: {best_acc:.2f}%")
    return best_acc, best_state

def train_main():
    """主训练函数"""
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
    
    model.print_trainable_params()
    
    # 阶段1: 训练分类头
    best_overall_acc = 0.0
    best_overall_state = None

    try:
        if config.stage1_epochs > 0:
            stage1_acc, stage1_state = train_stage1(
                model, train_loader, val_loader, device, config, save_dir, logger
            )
            best_overall_acc = stage1_acc
            best_overall_state = stage1_state
    
    # 阶段2: 微调后几层
        if config.stage2_epochs > 0:
            stage2_acc, stage2_state = train_stage2(
                model, train_loader, val_loader, device, config, save_dir, logger
            )
            if stage2_acc >= best_overall_acc:
                best_overall_acc = stage2_acc
                best_overall_state = stage2_state
    except KeyboardInterrupt:
        logger("\nTraining interrupted by user (KeyboardInterrupt). Saving best model so far...")
    finally:
        # 在训练结束或中断时统一保存一次最佳模型
        if best_overall_state is not None:
            save_checkpoint(
                state={
                    'epoch': 0,
                    'model_state_dict': best_overall_state,
                    'best_acc': best_overall_acc,
                    'config': config
                },
                save_dir=save_dir,
                filename='best_model.pth'
            )
            logger(f"✓ Best model saved to: {os.path.join(save_dir, 'best_model.pth')} (Acc@1: {best_overall_acc:.2f}%)")

    logger("\n" + "=" * 60)
    logger("Training Completed!")
    logger("=" * 60)
    logger(f"Results saved to: {save_dir}")
    logger(f"Best model: {os.path.join(save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    train_main()