from svm.Classifier import SVMClassifier
from dataloader.Dataloader import DataLoader
import random

def svm_method(
    data_root: str = './data',
    *,
    kernel: str = 'linear',
    C: float = 1.0,
    gamma: str | float = 'scale',
    degree: int = 3,
    epsilon: float = 1e-5,
    num_classes: int = 10,
    seed: int = 42,
):
    """
    运行 SVM 

    参数说明：
    - data_root: 数据集根目录。
    - kernel: 核函数类型，'linear' | 'rbf' | 'poly'。
    - C: 正则化系数。
    - gamma: 'scale' 或数值（rbf/poly 使用）。
    - degree: 多项式核的次数。
    - epsilon: 支持向量筛选阈值
    - num_classes: 随机抽取用于训练评估的类别数。
    - seed
    """

    # 解析 gamma：支持传入字符串 'scale' 或可解析为 float 的字符串/数值
    gamma_val: str | float
    if isinstance(gamma, str):
        if gamma.strip().lower() == 'scale':
            gamma_val = 'scale'
        else:
            try:
                gamma_val = float(gamma)
            except ValueError:
                raise ValueError(f"gamma 参数不合法：{gamma}，应为 'scale' 或数值")
    else:
        gamma_val = float(gamma)

    train_loader = DataLoader(data_root, split='train', mode='feature')

    # 类别采样（类别 id 从 1 开始，保持与 DataLoader 的标签映射一致）
    total_classes = len(train_loader.class_dirs)
    if num_classes < 1 or num_classes > total_classes:
        raise ValueError(f"num_classes 应在 [1, {total_classes}]，当前为 {num_classes}")
    random.seed(seed)
    target_labels = random.sample(range(1, total_classes + 1), num_classes)

    # 构建分类器
    classifier = SVMClassifier(
        kernel=kernel,
        C=C,
        gamma=gamma_val,
        degree=degree,
        epsilon=epsilon,
    )
    print(f"分类器参数: kernel={classifier.kernel}, C={classifier.C}")
    print(f"使用类别数: {num_classes}")

    # 训练集
    train_features, train_labels = train_loader.get_data_and_labels(
        target_labels=target_labels
    )
    classifier.train(train_features, train_labels)

    # 验证集
    val_loader = DataLoader(data_root, split='val', mode='feature')
    val_features, val_labels = val_loader.get_data_and_labels(target_labels=target_labels)
    val_accuracy = classifier.evaluate(val_features, val_labels)
    print(f"验证集准确率: {val_accuracy * 100:.2f}%")
