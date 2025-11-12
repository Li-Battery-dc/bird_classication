"""
ViT 可视化工具
功能：
1. 可视化训练日志中的loss和准确率曲线
2. 可视化指定图像的attention map
"""

import os
import re
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from .modules.ViT_model import create_vit_base_patch16


def parse_log_file(log_path):
    """
    解析训练日志文件
    
    Args:
        log_path: 日志文件路径
        
    Returns:
        dict: 包含训练和验证数据的字典
    """
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式提取数据
    # 匹配格式: Epoch 31/260 [Stage 2]
    # Train - Loss: 4.8913, Acc@1: 5.03%, Acc@5: 16.47%
    # Val   - Loss: 4.5237, Acc@1: 9.38%, Acc@5: 25.67%
    
    pattern = r'Epoch (\d+)/\d+.*?\nTrain - Loss: ([\d.]+), Acc@1: ([\d.]+)%.*?\nVal   - Loss: ([\d.]+), Acc@1: ([\d.]+)%'
    matches = re.findall(pattern, content)
    
    for match in matches:
        epoch, t_loss, t_acc, v_loss, v_acc = match
        epochs.append(int(epoch))
        train_loss.append(float(t_loss))
        train_acc.append(float(t_acc))
        val_loss.append(float(v_loss))
        val_acc.append(float(v_acc))
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }


def plot_training_curves(log_path, save_dir=None):
    """
    绘制训练曲线
    
    Args:
        log_path: 训练日志路径
        save_dir: 保存目录，默认保存到日志同目录
    """
    print(f"📊 解析训练日志: {log_path}")
    data = parse_log_file(log_path)
    
    if len(data['epochs']) == 0:
        print("❌ 未能从日志文件中提取到数据")
        return
    
    print(f"✓ 成功提取 {len(data['epochs'])} 个epoch的数据")
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 绘制Loss曲线
    axes[0].plot(data['epochs'], data['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(data['epochs'], data['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 绘制Accuracy曲线
    axes[1].plot(data['epochs'], data['train_acc'], label='Train Acc@1', linewidth=2)
    axes[1].plot(data['epochs'], data['val_acc'], label='Val Acc@1', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    if save_dir is None:
        save_dir = os.path.dirname(log_path)
    
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存到: {save_path}")
    
    # 显示统计信息
    print(f"\n📈 训练统计:")
    print(f"  最佳验证准确率: {max(data['val_acc']):.2f}% (Epoch {data['epochs'][np.argmax(data['val_acc'])]})")
    print(f"  最低验证Loss: {min(data['val_loss']):.4f} (Epoch {data['epochs'][np.argmin(data['val_loss'])]})")
    print(f"  最终验证准确率: {data['val_acc'][-1]:.2f}%")


def visualize_attention(image_path, model_path, layer_idx=-1, save_dir=None):
    """
    可视化指定图像的attention map
    
    Args:
        image_path: 输入图像路径
        model_path: 模型权重路径
        config_path: 配置文件路径（可选）
        layer_idx: Transformer层索引，-1表示最后一层
        save_dir: 保存目录，默认为result_dir
    """
    print(f"\n🔍 可视化Attention Map")
    print(f"  图像: {image_path}")
    print(f"  模型: {model_path}")
    print(f"  Layer: {layer_idx}")
    
    # 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_vit_base_patch16(config=None) # 创建默认的空模型
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("✓ 模型加载成功")
    
    # 加载和预处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 获取attention maps
    with torch.no_grad():
        attention_maps = model.get_attention_maps(img_tensor, layer_idx=layer_idx)
    
    if attention_maps is None:
        print("❌ 无法获取attention maps")
        return
    
    # attention_maps shape: (1, num_heads, num_patches+1, num_patches+1)
    attention_maps = attention_maps[0].cpu().numpy()  # (num_heads, 197, 197)
    num_heads = attention_maps.shape[0]
    
    # 提取CLS token对所有patch的attention (第一行，跳过CLS自己)
    cls_attention = attention_maps[:, 0, 1:]  # (num_heads, 196)
    
    # 计算平均attention
    avg_attention = cls_attention.mean(axis=0)  # (196,)
    
    # Reshape到2D grid
    num_patches = int(np.sqrt(cls_attention.shape[1]))
    cls_attention_2d = cls_attention.reshape(num_heads, num_patches, num_patches)
    avg_attention_2d = avg_attention.reshape(num_patches, num_patches)
    
    # 创建可视化
    fig = plt.figure(figsize=(16, 10))
    
    # 显示原图
    ax = plt.subplot(3, 5, 1)
    ax.imshow(img)
    ax.set_title('Original Image', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 显示平均attention
    ax = plt.subplot(3, 5, 2)
    im = ax.imshow(avg_attention_2d, cmap='jet', interpolation='bilinear')
    ax.set_title('Average Attention', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 显示叠加后的图像
    ax = plt.subplot(3, 5, 3)
    img_resized = np.array(img.resize((num_patches * 16, num_patches * 16)))
    attention_upsampled = np.array(Image.fromarray(
        (avg_attention_2d * 255).astype(np.uint8)
    ).resize((num_patches * 16, num_patches * 16), Image.BILINEAR))
    attention_upsampled = attention_upsampled / 255.0
    
    ax.imshow(img_resized)
    ax.imshow(attention_upsampled, cmap='jet', alpha=0.5, interpolation='bilinear')
    ax.set_title('Attention Overlay', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 显示各个head的attention
    for i in range(min(12, num_heads)):
        ax = plt.subplot(3, 5, i + 4)
        im = ax.imshow(cls_attention_2d[i], cmap='jet', interpolation='bilinear')
        ax.set_title(f'Head {i+1}', fontsize=10)
        ax.axis('off')
    
    layer_name = f"Layer {layer_idx}" if layer_idx >= 0 else f"Layer {model.depth + layer_idx}"
    plt.suptitle(f'Attention Visualization - {layer_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存
    if save_dir is None:
        save_dir = './result/vit/vis_images'
    
    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f'attention_{img_name}_layer{layer_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Attention可视化已保存到: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='ViT 可视化工具')
    parser.add_argument('--mode', type=str, required=True, choices=['curves', 'attention'],
                        help='可视化模式: curves(训练曲线) 或 attention(注意力图)')
    
    # 训练曲线参数
    parser.add_argument('--log', type=str, help='训练日志文件路径')
    
    # Attention可视化参数
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--model', type=str, help='模型权重路径')
    parser.add_argument('--layer', type=int, default=-1, help='Transformer层索引 (-1表示最后一层)')
    
    # 通用参数
    parser.add_argument('--save_dir', type=str, default=None, help='保存目录')
    
    args = parser.parse_args()
    
    if args.mode == 'curves':
        if not args.log:
            print("❌ 请指定训练日志文件 (--log)")
            return
        if not os.path.exists(args.log):
            print(f"❌ 日志文件不存在: {args.log}")
            return
        plot_training_curves(args.log, args.save_dir)
        
    elif args.mode == 'attention':
        if not args.image or not args.model:
            print("❌ 请指定图像路径 (--image) 和模型路径 (--model)")
            return
        if not os.path.exists(args.image):
            print(f"❌ 图像文件不存在: {args.image}")
            return
        if not os.path.exists(args.model):
            print(f"❌ 模型文件不存在: {args.model}")
            return
        visualize_attention(args.image, args.model, args.layer, args.save_dir)


if __name__ == '__main__':
    main()
