from Classifier import SVMClassifier
from dataloader.Dataloader import DataLoader
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

def test_kernel_with_different_C():
    """测试不同核函数在不同C值下的性能"""
    print("\n" + "="*70)
    print("测试不同核函数在不同C值下的性能")
    print("="*70)
    
    data_root = '/home/stu12/homework/MLPR/data'
    train_loader = DataLoader(data_root, split='train', mode='feature')
    
    random.seed(42)
    target_labels = random.sample(range(1, len(train_loader.class_dirs)+1), 10)
    train_features, train_labels = train_loader.get_data_and_labels(target_labels=target_labels)
    
    val_loader = DataLoader(data_root, split='val', mode='feature')
    val_features, val_labels = val_loader.get_data_and_labels(target_labels=target_labels)
    
    # 测试配置
    kernels = ['linear', 'rbf', 'poly']
    C_values = [0.1, 1, 10, 100]
    
    results = {kernel: {'C_values': C_values, 'train_accs': [], 'val_accs': []} 
               for kernel in kernels}
    
    for kernel in kernels:
        print(f"\n=== 测试 {kernel} 核函数 ===")
        for C in C_values:
            print(f"  C={C}...", end=" ")
            classifier = SVMClassifier(kernel=kernel, C=C)
            classifier.train(train_features, train_labels)
            
            train_acc = classifier.evaluate(train_features, train_labels)
            val_acc = classifier.evaluate(val_features, val_labels)
            
            results[kernel]['train_accs'].append(train_acc)
            results[kernel]['val_accs'].append(val_acc)
            
            print(f"验证集准确率: {val_acc * 100:.2f}%")
    
    return results

def plot_kernel_C_curves(results, save_path='/home/stu12/homework/MLPR/result/svm/'):
    """绘制不同核函数在不同C值下的曲线对比"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    markers = ['o', 's', '^', 'D']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (kernel, color, marker) in enumerate(zip(results.keys(), colors, markers)):
        C_values = results[kernel]['C_values']
        train_accs = [acc * 100 for acc in results[kernel]['train_accs']]
        val_accs = [acc * 100 for acc in results[kernel]['val_accs']]
        
        # 训练集准确率
        ax1.plot(C_values, train_accs, marker=marker, linestyle='-', 
                linewidth=2, markersize=8, label=kernel, color=color)
        
        # 验证集准确率
        ax2.plot(C_values, val_accs, marker=marker, linestyle='-', 
                linewidth=2, markersize=8, label=kernel, color=color)
    
    # 设置训练集图
    ax1.set_xscale('log')
    ax1.set_xlabel('C Value (log scale)')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 设置验证集图
    ax2.set_xscale('log')
    ax2.set_xlabel('C Value (log scale)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('SVM Performance: Different Kernels with Varying C Values', 
                fontsize=14, y=1.02)
    plt.tight_layout()
    
    # 保存图片到指定路径
    save_file = os.path.join(save_path, 'kernel_C_curves.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {save_file}")
    plt.close()
    # plt.show()

def save_results_to_file(results, save_path='/home/stu12/homework/MLPR/result/svm/'):
    """将实验结果保存到文本文件"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    save_file = os.path.join(save_path, 'experiment_results.txt')
    
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("SVM 参数实验结果\n")
        f.write("="*70 + "\n\n")
        
        # 找出最佳参数
        best_val_acc = 0
        best_config = None
        
        for kernel in results.keys():
            C_values = results[kernel]['C_values']
            train_accs = results[kernel]['train_accs']
            val_accs = results[kernel]['val_accs']
            
            f.write(f"\n{'='*50}\n")
            f.write(f"核函数: {kernel}\n")
            f.write(f"{'='*50}\n")
            
            for i, C in enumerate(C_values):
                train_acc = train_accs[i] * 100
                val_acc = val_accs[i] * 100
                
                f.write(f"\nC = {C:>6}\n")
                f.write(f"  训练集准确率: {train_acc:6.2f}%\n")
                f.write(f"  验证集准确率: {val_acc:6.2f}%\n")
                
                # 更新最佳配置
                if val_accs[i] > best_val_acc:
                    best_val_acc = val_accs[i]
                    best_config = {
                        'kernel': kernel,
                        'C': C,
                        'train_acc': train_accs[i],
                        'val_acc': val_accs[i]
                    }
        
        # 写入最佳配置
        f.write(f"\n\n{'='*70}\n")
        f.write("最佳参数配置\n")
        f.write(f"{'='*70}\n")
        f.write(f"核函数: {best_config['kernel']}\n")
        f.write(f"C 值: {best_config['C']}\n")
        f.write(f"训练集准确率: {best_config['train_acc'] * 100:.2f}%\n")
        f.write(f"验证集准确率: {best_config['val_acc'] * 100:.2f}%\n")
    
    print(f"\n实验结果已保存: {save_file}")
    print(f"\n最佳配置: kernel={best_config['kernel']}, C={best_config['C']}, "
          f"验证集准确率={best_config['val_acc']*100:.2f}%")
    
    return best_config


def plot_kernel_C_heatmap(results, save_path='/home/stu12/homework/MLPR/result/svm/'):
    """绘制核函数与C值的热力图"""
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)
    
    kernels = list(results.keys())
    C_values = results[kernels[0]]['C_values']
    
    # 创建验证集准确率矩阵
    val_acc_matrix = np.array([results[k]['val_accs'] for k in kernels]) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(val_acc_matrix, cmap='YlOrRd', aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(C_values)))
    ax.set_yticks(np.arange(len(kernels)))
    ax.set_xticklabels(C_values)
    ax.set_yticklabels(kernels)
    
    ax.set_xlabel('C Value')
    ax.set_ylabel('Kernel Function')
    ax.set_title('Validation Accuracy (%) for Different Kernels and C Values')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)
    
    # 在每个格子中显示数值
    for i in range(len(kernels)):
        for j in range(len(C_values)):
            text = ax.text(j, i, f'{val_acc_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    plt.tight_layout()
    
    # 保存图片到指定路径
    save_file = os.path.join(save_path, 'kernel_C_heatmap.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存: {save_file}")
    plt.close()
    # plt.show()

if __name__ == "__main__":
    # 设置保存路径
    save_path = '/home/stu12/homework/MLPR/result/svm/'
    
    # 运行实验
    print("开始运行SVM参数实验...")
    combined_results = test_kernel_with_different_C()
    
    # 保存结果到文件
    best_config = save_results_to_file(combined_results, save_path)
    
    # 生成可视化图表
    print("\n生成可视化图表...")
    plot_kernel_C_heatmap(combined_results, save_path)
    plot_kernel_C_curves(combined_results, save_path)
    
    print("\n" + "="*70)
    print("所有实验完成！")
    print(f"结果保存在: {save_path}")
    print("="*70)