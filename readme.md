# 鸟类识别分类任务

本项目针对 CUB-200-2011 鸟类数据集，提供基于 CNN 的深度学习分类器与基于 SVM 的传统机器学习基线，并配备统一的命令行接口、数据加载器与训练日志可视化工具。本文档聚焦 SVM 与 CNN 的功能模块，暂不覆盖 ViT 相关内容。

## 环境准备

- Python ≥ 3.9
- 安装依赖：`pip install -r requirements.txt`
- 推荐使用 GPU 进行 CNN 训练（脚本会自动检测 `cuda`），但 SVM 与可视化均可在 CPU 上运行。

## 目录总览

- `src/dataloader/` 数据加载器，支持图像模式与预提取特征模式，提供 `get_data_and_labels` 等便捷接口。
- `src/cnn/` CNN 模块，包含网络结构、训练脚本、验证脚本与可视化工具。
- `src/svm/` SVM 模块，封装了基于 sklearn 的分类器。
- `result/` 默认的权重、日志、可视化输出目录。
- `data/` 默认的数据集放置路径，包含 `train/` 与 `val/` 子目录。

## 数据准备

请确保数据集目录结构

```
data/
	train/
		001.Black_footed_Albatross/
		...
	val/
		001.Black_footed_Albatross/
		...
```

数据加载器支持两种模式：
- `mode='image'` 逐图像加载并应用数据增强（CNN 使用）。
- `mode='feature'` 加载 `.pt` 特征文件（SVM 使用）。

## 入口脚本 `src/main.py`

`main.py` 已重构为统一的命令行入口，支持以下子命令：

- `cnn-train` 训练 CNN 分类器。
- `cnn-validate` 在验证集上评估已训练权重或 checkpoint。
- `cnn-visualize` 汇总训练日志与 checkpoint 指标并生成图表。
- `svm` 运行 SVM 基线实验。

运行方式（工程根目录）：

```bash
python src/main.py <sub-command> [options]
```

## SVM 方法

SVM 使用 `dataloader.DataLoader` 的特征模式载入预提取特征，默认随机抽取 10 个类别并训练线性核分类器，随后在同一子集上进行验证并打印准确率。

子命令与参数：

```bash
python src/main.py svm \
	--data-root /home/stu12/homework/MLPR/data \
	[--kernel {linear,rbf,poly}] \
	[--C 1.0] \
	[--gamma scale|<float>] \
	[--degree 3] \
	[--epsilon 1e-5] \
	[--num-classes 10] \
	[--seed 42]
```

参数说明：

- `--kernel`：核函数类型，支持 `linear`、`rbf`、`poly`，默认 `linear`。
- `--C`：正则化系数，默认 `1.0`。
- `--gamma`：`rbf`/`poly` 的核宽度；可为 `scale`（默认）或显式数值（如 `0.1`）。
- `--degree`：多项式核次数，默认 `3`。
- `--epsilon`：用于数值稳定的支持向量筛选阈值，默认 `1e-5`。
- `--num-classes`：从全部类别中随机抽取的子集大小，默认 `10`。
- `--seed`：类别采样的随机种子，默认 `42`。

示例：

- RBF 核并指定 `gamma=0.1`、`C=4.0`，并使用 20 个类别：

```bash
python src/main.py svm \
	--data-root /home/stu12/homework/MLPR/data \
	--kernel rbf \
	--gamma 0.1 \
	--C 4.0 \
	--num-classes 10 \
	--seed 42
```

## CNN 训练

`cnn-train` 子命令封装了 `cnn.Method.train`。核心参数：

- `--weight-save-path` 最终权重保存路径（必填）。
- `--ckpt-load-path` 断点续训时指定已有 checkpoint。
- `--data-root` 数据集根目录，默认 `/home/stu12/homework/MLPR/data/`。
- `--freeze / --no-freeze` 控制渐进式冻结策略，默认开启。
- `--freeze-epoch-ratio` 冻结开始的 epoch 比例，默认 0.75。
- `--warmup / --no-warmup` 控制 warmup 学习率调度，默认开启。
- `--warmup-ratio` warmup 占用的 epoch 比例，默认 0.1。
- `--num-epochs` 训练轮数，默认 1000。
- `--batch-size` batch 大小，默认 512。

示例命令：

```bash
python src/main.py cnn-train \
	--weight-save-path result/cnn/weights/resnet_latest.pth \
	--data-root /home/stu12/homework/MLPR/data/ \
	--num-epochs 800 \
	--batch-size 512
```

训练脚本会：
- 自动创建日志目录 `result/cnn/logs/`，文件名形如 `train_log_YYYYMMDD_HHMMSS.txt`。
- 每 50 轮保存一次 checkpoint 至 `result/cnn/ckpts/train_时间戳/`。

## CNN 验证

`cnn-validate` 子命令封装 `cnn.Method.validate`，用于对成品权重或 checkpoint 进行验证。

示例命令：

```bash
python src/main.py cnn-validate \
	--weight-path result/cnn/weights/resnet_latest.pth \
	--data-root /home/stu12/homework/MLPR/data/ \
	--num-classes 200
```

若指定的是 checkpoint（包含 `model_state_dict`），脚本会自动解析并加载模型参数。

## CNN 训练可视化

`cnn-visualize` 子命令调用 `cnn.visulize_utils.visualizes_all`，整合以下能力：

- 解析训练日志并绘制 Loss / Accuracy 曲线。
- 批量评估 checkpoint，绘制验证集指标曲线。
- 对比训练与验证曲线，并按训练时间戳分文件夹保存。

示例命令：

```bash
python src/main.py cnn-visualize \
	--log-file result/cnn/logs/train_log_20251016_172005.txt \
	--ckpts-dir result/cnn/ckpts/train_20251016_172005 \
	--result-dir result/cnn \
	--data-root /home/stu12/homework/MLPR/data
```

输出图像保存在 `result_dir/vis_images/train_时间戳/`