import os
import re
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from .validate import validate_model
from dataloader.Dataloader import DataLoader
from torchvision import transforms


def extract_train_timestamp(path):
    """
    从路径中提取训练时间戳标识符
    例如：从 'result/cnn/ckpts/train_20251012_162440' 提取 'train_20251012_162440'
    
    Args:
        path: 包含训练时间戳的路径
        
    Returns:
        训练时间戳字符串，如果未找到则返回 None
    """
    match = re.search(r'(train_\d{8}_\d{6})', path)
    if match:
        return match.group(1)
    return None


def parse_training_log(log_file_path):

    epochs = []
    losses = []
    accuracies = []
    
    with open(log_file_path, 'r') as f:
        for line in f:
            # 匹配格式：Epoch [1/100], time: 2025-10-12 13:14:32
            epoch_match = re.search(r'Epoch \[(\d+)/\d+\]', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                epochs.append(epoch)
            
            # 匹配格式：Loss: 5.2959, Accuracy: 0.80%
            metrics_match = re.search(r'Loss: ([\d.]+), Accuracy: ([\d.]+)%', line)
            if metrics_match:
                loss = float(metrics_match.group(1))
                accuracy = float(metrics_match.group(2))
                losses.append(loss)
                accuracies.append(accuracy)
    
    return epochs, losses, accuracies


def visualize_training_log(log_file_path, save_dir=None):
    
    epochs, losses, accuracies = parse_training_log(log_file_path)
    
    if len(epochs) == 0:
        print(f"No training data found in {log_file_path}")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制 Loss 曲线
    ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制 Accuracy 曲线
    ax2.plot(epochs, accuracies, 'r-', linewidth=2, label='Training Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy over Epochs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    if save_dir is not None:
        # 从日志文件路径中提取训练时间戳
        train_timestamp = extract_train_timestamp(log_file_path)
        
        if train_timestamp:
            # 创建带时间戳的子目录
            save_dir = os.path.join(save_dir, train_timestamp)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, 'training_log_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training visualization saved to {save_path}")
    
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Training Statistics:")
    print("="*60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f}")
    print(f"Initial Accuracy: {accuracies[0]:.2f}% -> Final Accuracy: {accuracies[-1]:.2f}%")
    print(f"Best Accuracy: {max(accuracies):.2f}% (Epoch {epochs[accuracies.index(max(accuracies))]})")
    print(f"Lowest Loss: {min(losses):.4f} (Epoch {epochs[losses.index(min(losses))]})")
    print("="*60)


def validation_ckpt_and_visulize(ckpts_dir, data_root, num_classes=200, save_dir=None):
    """
    评估指定目录下所有 checkpoint 在验证集上的准确率，并可视化
    
    Args:
        ckpts_dir: checkpoint 文件所在目录
        data_root: 数据集根目录
        num_classes: 类别数量
        save_dir: 保存图片的目录，如果为 None 则只显示不保存
    """
    # 验证集的 transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    
    # 加载验证数据集
    print("Loading validation dataset...")
    val_loader = DataLoader(data_root, split='val', mode='image', transform=val_transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    
    # 查找所有 checkpoint 文件
    ckpt_files = []
    if os.path.exists(ckpts_dir):
        for filename in os.listdir(ckpts_dir):
            if filename.endswith('.pth') and 'checkpoint_epoch' in filename:
                ckpt_files.append(filename)
    
    if len(ckpt_files) == 0:
        print(f"No checkpoint files found in {ckpts_dir}")
        return
    
    # 按照 epoch 排序
    def extract_epoch(filename):
        match = re.search(r'checkpoint_epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    ckpt_files.sort(key=extract_epoch)
    
    # 评估每个 checkpoint
    epochs = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nEvaluating {len(ckpt_files)} checkpoints...")
    print("="*60)
    
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpts_dir, ckpt_file)
        epoch = extract_epoch(ckpt_file)
        
        print(f"Evaluating {ckpt_file}...")
        
        try:
            # 直接传递 checkpoint 路径给 validate_model
            val_loss, val_acc = validate_model(
                val_loader, 
                criterion, 
                device, 
                state_dict_path=ckpt_path,  # 传递文件路径
                num_classes=num_classes
            )
            
            epochs.append(epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc * 100)  # 转换为百分比
            
            print(f"  Epoch {epoch}: Loss={val_loss:.4f}, Accuracy={val_acc*100:.2f}%")
        
        except Exception as e:
            print(f"  Error evaluating {ckpt_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("="*60)
    
    if len(epochs) == 0:
        print("No valid checkpoint evaluations")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制验证 Loss 曲线
    ax1.plot(epochs, val_losses, 'b-o', linewidth=2, markersize=6, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Validation Loss over Checkpoints', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制验证 Accuracy 曲线
    ax2.plot(epochs, val_accuracies, 'r-o', linewidth=2, markersize=6, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy over Checkpoints', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    if save_dir is not None:
        # 从 checkpoint 目录路径中提取训练时间戳
        train_timestamp = extract_train_timestamp(ckpts_dir)
        
        if train_timestamp:
            # 创建带时间戳的子目录
            save_dir = os.path.join(save_dir, train_timestamp)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, 'checkpoint_validation_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCheckpoint validation visualization saved to {save_path}")

    # plt.show()

    # 打印统计信息
    print("\n" + "="*60)
    print("Validation Statistics:")
    print("="*60)
    print(f"Total Checkpoints Evaluated: {len(epochs)}")
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}% (Epoch {epochs[val_accuracies.index(max(val_accuracies))]})")
    print(f"Lowest Validation Loss: {min(val_losses):.4f} (Epoch {epochs[val_losses.index(min(val_losses))]})")
    print("="*60)


def compare_training_and_validation(log_file_path, ckpts_dir, data_root, num_classes=200, save_dir=None):
    """
    对比训练和验证的 loss 和 accuracy 曲线
    
    Args:
        log_file_path: 训练日志文件路径
        ckpts_dir: checkpoint 文件所在目录
        data_root: 数据集根目录
        num_classes: 类别数量
        save_dir: 保存图片的目录
    """
    # 解析训练日志
    train_epochs, train_losses, train_accuracies = parse_training_log(log_file_path)
    
    # 验证集的 transform
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])
    
    # 加载验证数据集
    print("Loading validation dataset...")
    val_loader = DataLoader(data_root, split='val', mode='image', transform=val_transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()
    
    # 查找所有 checkpoint 文件
    ckpt_files = []
    if os.path.exists(ckpts_dir):
        for filename in os.listdir(ckpts_dir):
            if filename.endswith('.pth') and 'checkpoint_epoch' in filename:
                ckpt_files.append(filename)
    
    # 按照 epoch 排序
    def extract_epoch(filename):
        match = re.search(r'checkpoint_epoch_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    ckpt_files.sort(key=extract_epoch)
    
    # 评估每个 checkpoint
    val_epochs = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nEvaluating {len(ckpt_files)} checkpoints...")
    
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpts_dir, ckpt_file)
        epoch = extract_epoch(ckpt_file)
        
        print(f"Evaluating {ckpt_file}...")
        
        try:
            # 直接传递 checkpoint 路径给 validate_model
            val_loss, val_acc = validate_model(
                val_loader, 
                criterion, 
                device, 
                state_dict_path=ckpt_path,  # 传递文件路径
                num_classes=num_classes
            )
            
            val_epochs.append(epoch)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc * 100)
        
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 绘制 Loss 对比曲线
    # 训练集：每个 epoch 都有数据，使用细线
    ax1.plot(train_epochs, train_losses, 'b-', linewidth=1.5, label='Training Loss', alpha=0.6)
    # 验证集：每 100 个 epoch 有数据，使用粗线和标记点
    if len(val_epochs) > 0:
        ax1.plot(val_epochs, val_losses, 'r-o', linewidth=2.5, markersize=8, label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 绘制 Accuracy 对比曲线
    # 训练集：每个 epoch 都有数据，使用细线
    ax2.plot(train_epochs, train_accuracies, 'b-', linewidth=1.5, label='Training Accuracy', alpha=0.6)
    # 验证集：每 100 个 epoch 有数据，使用粗线和标记点
    if len(val_epochs) > 0:
        ax2.plot(val_epochs, val_accuracies, 'r-o', linewidth=2.5, markersize=8, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # 保存图片
    if save_dir is not None:
        # 从 checkpoint 目录或日志文件路径中提取训练时间戳
        train_timestamp = extract_train_timestamp(ckpts_dir)
        if not train_timestamp:
            train_timestamp = extract_train_timestamp(log_file_path)
        
        if train_timestamp:
            # 创建带时间戳的子目录
            save_dir = os.path.join(save_dir, train_timestamp)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, 'training_vs_validation_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison visualization saved to {save_path}")

    # plt.show()


def visualizes_all(log_file, ckpts_dir,
        result_dir = "/home/stu12/homework/MLPR/result/cnn",
        data_root = "/home/stu12/homework/MLPR/data"):
    
    # 基础可视化保存目录
    vis_base_dir = os.path.join(result_dir, "vis_images")
    
    # 1. 可视化训练日志
    if os.path.exists(log_file):
        print("Visualizing training log...")
        visualize_training_log(log_file, save_dir=vis_base_dir)
    
    # 2. 可视化 checkpoint 验证结果
    if os.path.exists(ckpts_dir):
        print("\nVisualizing checkpoint validation...")
        validation_ckpt_and_visulize(ckpts_dir, data_root, num_classes=200, 
                                       save_dir=vis_base_dir)
    
    # 3. 对比训练和验证曲线
    if os.path.exists(log_file) and os.path.exists(ckpts_dir):
        print("\nComparing training and validation...")
        compare_training_and_validation(log_file, ckpts_dir, data_root, num_classes=200,
                                       save_dir=vis_base_dir)
