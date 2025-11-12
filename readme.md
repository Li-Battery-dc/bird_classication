# 鸟类识别分类任务

本项目针对 CUB-200-2011 鸟类数据集，实现了三种分类方法：
1. **CNN (ResNet)** - 自己设计和实现的深度学习分类器
2. **SVM** - 基于传统机器学习的分类器
3. **ViT (Vision Transformer)** - 基于 timm 预训练模型的微调方法

## 项目目录结构

```
bird_classification/
├── readme.md                       # 项目说明文档
├── requirements.txt                # 项目依赖
├── 课程大作业说明.pdf               # 作业说明
├── data/                           # 数据集目录
│   ├── train/                      # 训练集（200个鸟类子目录）
│   │   ├── 001.Black_footed_Albatross/
│   │   ├── 002.Laysan_Albatross/
│   │   └── ...
│   └── val/                        # 验证集（200个鸟类子目录）
│       └── ...
├── src/                            # 源代码目录
│   ├── main.py                     # 统一命令行入口
│   ├── dataloader/                 # 数据加载模块
│   │   ├── DataLoader.py           # 数据加载器（支持图像/特征模式）
│   │   └── ...
│   ├── cnn/                        # CNN模块
│   │   ├── Method.py               # 训练和验证方法
│   │   ├── network.py              # ResNet网络结构
│   │   ├── visulize_utils.py       # 可视化工具
│   │   └── ...
│   ├── svm/                        # SVM模块
│   │   ├── Method.py               # SVM分类器封装
│   │   └── ...
│   └── ViT/                        # ViT模块
│       ├── config.py               # 训练配置（硬编码参数）
│       ├── visualize.py            # 可视化脚本
│       ├── modules/                # ViT模型定义
│       │   ├── ViT_model.py        # ViT模型实现
│       │   └── ...
│       └── utils/                  # 工具函数
│           └── ...
└── result/                         # 结果输出目录
    ├── cnn/                        # CNN结果
    │   ├── weights/                # 模型权重
    │   ├── logs/                   # 训练日志
    │   ├── ckpts/                  # checkpoints
    │   └── vis_images/             # 可视化图像
    ├── svm/                        # SVM结果
    └── vit/                        # ViT结果
        └── train_TIMESTAMP/        # 每次训练的目录
            ├── train.log           # 训练日志
            ├── training_curves.png # 训练曲线
            ├── ckpt/               # checkpoints和模型
			└── config.txt          # 当次训练的config内容，当前的结果是手动保存的。
```

## 环境准备

- Python ≥ 3.9
- 安装依赖：`pip install -r requirements.txt`

## 数据准备

请确保数据集目录结构如下：

```
data/
├── train/
│   ├── 001.Black_footed_Albatross/
│   ├── 002.Laysan_Albatross/
│   └── ...
└── val/
    ├── 001.Black_footed_Albatross/
    └── ...
```

数据加载器支持两种模式：
- `mode='image'`：逐图像加载并应用数据增强（CNN 和 ViT 使用）
- `mode='feature'`：加载 `.pt` 特征文件（SVM 使用）

## 命令行入口

`src/main.py` 是统一的命令行入口，支持以下子命令：

- `svm` - 运行 SVM 基线实验
- `cnn-train` - 训练 CNN 分类器
- `cnn-validate` - 验证 CNN 模型
- `cnn-visualize` - 可视化 CNN 训练过程
- `vit-train` - 训练 ViT 分类器
- `vit-visualize` - 可视化 ViT 训练/Attention

基本用法：

```bash
python src/main.py <sub-command> [options]
```

查看帮助：

```bash
python src/main.py --help
python src/main.py <sub-command> --help
```

## SVM 方法

SVM 使用预提取特征进行训练，默认随机抽取指定数量的类别，训练线性核分类器并在验证集上评估。

### 使用方法

```bash
python src/main.py svm \
    --data-root ./data \
    [--kernel {linear,rbf,poly}] \
    [--C 1.0] \
    [--gamma scale|<float>] \
    [--degree 3] \
    [--epsilon 1e-5] \
    [--num-classes 10] \
    [--seed 42]
```

### 参数说明

- `--data-root`：数据集根目录，默认 `./data`
- `--kernel`：核函数类型（`linear`、`rbf`、`poly`），默认 `linear`
- `--C`：正则化系数，默认 `1.0`
- `--gamma`：核宽度参数（`rbf`/`poly` 使用），可为 `scale` 或数值，默认 `scale`
- `--degree`：多项式核次数，默认 `3`
- `--epsilon`：支持向量筛选阈值，默认 `1e-5`
- `--num-classes`：随机抽取的类别数量，默认 `10`
- `--seed`：随机种子，默认 `42`


## CNN 方法

CNN 方法使用 ResNet 架构，支持完整的训练、验证和可视化流程。

### CNN 训练

训练 CNN 分类器，支持渐进式冻结、warmup 学习率调度等策略。

**使用方法：**

```bash
python src/main.py cnn-train \
    --weight-save-path result/cnn/weights/resnet_latest.pth \
    --data-root ./data/ \
    [--ckpt-load-path <checkpoint_path>] \
    [--num-epochs 1000] \
    [--batch-size 512] \
    [--freeze / --no-freeze] \
    [--freeze-epoch-ratio 0.75] \
    [--warmup / --no-warmup] \
    [--warmup-ratio 0.1]
```

**参数说明：**

- `--weight-save-path`：最终权重保存路径（必填）
- `--ckpt-load-path`：从 checkpoint 恢复训练（可选）
- `--data-root`：数据集根目录，默认 `./data/`
- `--num-epochs`：训练轮数，默认 1000
- `--batch-size`：batch 大小，默认 512
- `--freeze` / `--no-freeze`：是否启用渐进式冻结策略，默认开启
- `--freeze-epoch-ratio`：冻结开始的 epoch 比例，默认 0.75
- `--warmup` / `--no-warmup`：是否启用 warmup 学习率调度，默认开启
- `--warmup-ratio`：warmup 占用的 epoch 比例，默认 0.1

**训练输出：**

- 训练日志保存在 `result/cnn/logs/train_log_YYYYMMDD_HHMMSS.txt`
- 每 50 轮保存一次 checkpoint 至 `result/cnn/ckpts/train_时间戳/`

**使用示例：**

```bash
python src/main.py cnn-train \
    --weight-save-path result/cnn/weights/resnet_latest.pth \
    --data-root ./data/ \
    --num-epochs 800 \
    --batch-size 512
```

### CNN 验证

在验证集上评估已训练的模型权重或 checkpoint。

**使用方法：**

```bash
python src/main.py cnn-validate \
    --weight-path result/cnn/weights/resnet_latest.pth \
    --data-root ./data/ \
    [--num-classes 200]
```

**参数说明：**

- `--weight-path`：模型权重或 checkpoint 路径（必填）
- `--data-root`：数据集根目录，默认 `./data/`
- `--num-classes`：类别数量，默认 200

脚本会自动识别权重格式（完整权重或 checkpoint）并加载。

### CNN 可视化

可视化 CNN 训练过程，顺序评估目录中所有的ckpts, 得到包括训练曲线评估曲线

**使用方法：**

```bash
python src/main.py cnn-visualize \
    --log-file result/cnn/logs/train_log_20251016_172005.txt \
    --ckpts-dir result/cnn/ckpts/train_20251016_172005 \
    --result-dir result/cnn \
    --data-root ./data
```

**参数说明：**

- `--log-file`：训练日志文件路径（必填）
- `--ckpts-dir`：checkpoints 目录路径（必填）
- `--result-dir`：结果保存目录（必填）
- `--data-root`：数据集根目录，默认 `./data`

**功能：**

- 解析训练日志并绘制 Loss / Accuracy 曲线
- 批量评估 checkpoints，绘制验证集指标曲线
- 对比训练与验证曲线

**输出：**

可视化图像保存在 `<result-dir>/vis_images/train_时间戳/`

## ViT (Vision Transformer) 方法

基于 timm 库预训练模型的 ViT 微调功能，采用多阶段渐进式训练策略。

### 配置说明

**重要**：ViT 的所有训练参数都在 `src/ViT/config.py` 中硬编码配置。**使用者需要直接修改该文件来调整参数**。

配置文件主要内容：

1. **模型架构配置**：
   - 图像大小、patch大小、embedding维度等
   - 默认使用 ViT-Base/16 架构，更改后加载预训练模型出错

2. **预训练配置**：
   - `use_pretrained`：是否使用预训练权重
   - `pretrained_model`：timm 模型名称（如 `'vit_base_patch16_224_in21k'`）
3. **数据配置**：`data_root`、`result_dir`、`batch_size_val` 等
4. **Checkpoint 恢复**：`resume_from_checkpoint`（设置路径恢复训练，`None` 从头开始）
5. **多阶段训练**：
   - **Stage 1**：只训练分类头（冻结 backbone）
   - **Stage 2**：解冻 backbone，较小学习率微调
   - **Stage 3**：(可选) 更小学习率进一步优化

### ViT 训练

**使用方法：**

```bash
python src/main.py vit-train
```

**训练参数通过修改 `src/ViT/config.py` 配置。**

**训练输出：**

- 自动创建带时间戳的保存目录（如 `result/vit/train_20251111_205138/`）
- 保存训练日志到 `train.log`
- 每个 stage 结束时保存 checkpoint
- 定期保存 checkpoint（每 10 个 epoch）
- 保存验证集最佳模型 `best_model.pth`

**训练策略：**

- 采用多阶段渐进式训练（Stage 1 → Stage 2 → Stage 3）
- 支持 Mixup/CutMix 数据增强
- 使用 CosineLR 学习率调度器
- 支持 Layer-wise Learning Rate Decay (LLRD)
- 自动保存最佳模型和定期 checkpoint

### ViT 可视化

ViT 提供两种可视化模式。

#### 1. 训练曲线可视化

绘制训练和验证集的 Loss、Accuracy 变化曲线。

**使用方法：**

```bash
python src/main.py vit-visualize curves \
    --log result/vit/train_20251111_205138/train.log \
    [--save-dir result/vit/train_20251111_205138]
```

**参数说明：**

- `--log`：训练日志文件路径（必填）
- `--save-dir`：保存目录（可选，默认与日志文件同目录）

**输出：**

- 生成 `training_curves.png` 包含 Loss 和 Accuracy 曲线
- 打印训练统计信息（最佳准确率、最低 Loss 等）

#### 2. Attention 可视化

可视化指定图像在特定 Transformer 层的注意力分布。

**使用方法：**

```bash
python src/main.py vit-visualize attention \
    --image data/val/177.Prothonotary_Warbler/Prothonotary_Warbler_0098_173913.jpg \
    --model result/vit/train_20251111_205138/ckpt/best_model.pth \
    [--layer -1] \
    [--save-dir result/vit/vis_images]
```

**参数说明：**

- `--image`：输入图像路径（必填）
- `--model`：模型 checkpoint 路径（必填）
- `--layer`：Transformer 层索引（默认 -1，表示最后一层；0-11 表示具体层）
- `--save-dir`：保存目录（默认：`result/vit/vis_images`）

**输出：**

生成包含以下内容的可视化图像：
- 原始图像
- 平均注意力图
- 注意力叠加图
- 各个 attention head 的注意力分布

### ViT 结果目录结构

训练后的目录结构示例：

```
result/vit/
└── train_20251111_205138/
    ├── train.log                    # 训练日志
    ├── training_curves.png          # 训练曲线（可视化后生成）
    └── ckpt/
        ├── checkpoint_epoch_10.pth  # 定期保存的 checkpoint
        ├── checkpoint_epoch_20.pth
        └── best_model.pth           # 验证集最佳模型
```