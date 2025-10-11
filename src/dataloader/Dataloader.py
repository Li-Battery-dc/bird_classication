import os
import os.path as osp
import torch
import numpy as np
from typing import Tuple, List, Optional, Union
from PIL import Image
import random


class DataLoader:
    """
    CUB-200-2011 鸟类数据集的数据加载器
    支持加载预提取的特征文件(.pt)和原始图像文件(.jpg)
    """
    
    def __init__(self, 
                 data_root: str, 
                 split: str = 'train',
                 mode: str = 'feature',
                 transform=None,
                 class_limit: Optional[int] = None):
        """
        初始化数据集
        
        Args:
            data_root: 数据集根目录路径
            split: 数据集分割 ('train' 或 'val')
            mode: 加载模式 ('feature', 'image')
            transform: 图像预处理变换
            class_limit: 限制加载的类别数量，用于快速测试
        """
        self.data_root = data_root
        self.split = split
        self.load_features = mode == 'feature'
        self.load_images = mode == 'image'
        self.transform = transform
        
        self.data_dir = osp.join(data_root, split)
        if not osp.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # load data dirs
        self.class_dirs = sorted([d for d in os.listdir(self.data_dir) 
                                 if osp.isdir(osp.join(self.data_dir, d))])
        
        if class_limit is not None:
            self.class_dirs = self.class_dirs[:class_limit]
        
        # map class names to indices
        self.class_to_label = {cls_name: (idx + 1) for idx, cls_name in enumerate(self.class_dirs)}
        self.label_to_class = {idx: cls_name for cls_name, idx in self.class_to_label.items()}
        
        # collect all data paths
        feature_samples = []
        image_samples = []
        
        for class_name in self.class_dirs:
            class_dir = osp.join(self.data_dir, class_name)
            label = self.class_to_label[class_name]
            
            # 获取该类别下的所有文件
            files = os.listdir(class_dir)
            
            # 找到所有的特征文件或图像文件
            if self.load_features:
                feature_files = [f for f in files if f.endswith('.pt')]
                for feature_file in feature_files:
                    feature_path = osp.join(class_dir, feature_file)
                    # 确保文件存在
                    if osp.exists(feature_path):
                        feature_samples.append((feature_path, label))
            if self.load_images:
                image_files = [f for f in files if f.endswith('.jpg')]
                for image_file in image_files:
                    image_path = osp.join(class_dir, image_file)
                    if osp.exists(image_path):
                        image_samples.append((image_path, label))

        self.samples = feature_samples if self.load_features else image_samples
        
        print("loaded dataset info:")
        self.show_info()
        print("-----------------------")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def load_sample(self, sample: Tuple[str, int]) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (data, label) where data can be features or image
        """
        data_path, label = sample
        
        data = None
        
        # 加载特征
        if self.load_features and data_path:
            try:
                features = torch.load(data_path, map_location='cpu')
                if isinstance(features, torch.Tensor):
                    data = features.numpy()  # 转换为numpy数组
                else:
                    data = np.array(features)

                # # 中位数填充nan值
                # median = np.nanmedian(data)
                # nan_idx = np.isnan(data)
                # data[nan_idx] = median
            except Exception as e:
                print(f"加载特征文件失败 {data_path}: {e}")
                data = np.zeros(384, dtype=np.float32)  # 默认特征维度
        
        # 加载图像
        if self.load_images and data_path:
            try:
                image = Image.open(data_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                else:
                    # 默认转换为numpy数组
                    image = np.array(image)

                # # 中位数填充nan值
                # for c in range(image.shape[2]):
                #     channel_data = image[:, :, c]
                #     median = np.nanmedian(channel_data)
                #     nan_idx = np.isnan(channel_data)
                #     channel_data[nan_idx] = median
                #     image[:, :, c] = channel_data

                data = image
            except Exception as e:
                print(f"加载图像文件失败 {data_path}: {e}")
                if data is None:
                    data = np.zeros((224, 224, 3), dtype=np.uint8)  # 默认图像大小
            
        return data, label
    
    def __getitem__(self, idx: int) -> Tuple[Union[torch.Tensor, np.ndarray], int]:
        return self.load_sample(self.samples[idx])

    def get_class_name(self, label: int) -> str:
        """根据类别索引获取类别名称"""
        return self.label_to_class.get(label, f"Unknown_{label}")

    def get_class_names(self) -> List[str]:
        """获取所有类别名称"""
        return self.class_dirs

    def get_data_and_labels(self, num_samples=None, target_labels=None, target_class_names=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量获取指定数据和标签
        param num_samples: 如果指定，则只加载该数量的样本（用于快速测试）
        param target_classes: 如果指定，则只加载这些类别的样本（类别索引列表）
        param target_class_names: 如果指定，则只加载这些类别的样本（类别名称列表）
        
        Returns:
            (data(image/feature), labels) 两个numpy数组
        """
        data_list = []
        labels_list = []
        filtered_samples = self.samples

        if target_labels is not None and target_class_names is not None:
            raise ValueError("只能指定 target_labels 或 target_class_names 之一")

        if target_class_names is not None:
            target_labels = [label for label in range(1, len(self.class_dirs)+1) if self.get_class_name(label) in target_class_names]
        if target_labels is not None:
            filtered_samples = [s for s in self.samples if s[1] in target_labels]

        if num_samples is not None:
            filtered_samples = random.sample(filtered_samples, min(num_samples, len(filtered_samples)))

        for sample in filtered_samples:
            data, label = self.load_sample(sample)
            data_list.append(data)
            labels_list.append(label)

        data_array = np.vstack(data_list)
        labels_array = np.array(labels_list)

        print("Loaded data and labels:")
        print(f"data size: {data_array.shape}")
        print(f"label size: {labels_array.shape}")
        print("\n")

        return data_array, labels_array
    
    def show_info(self):
        """打印数据集信息"""
        print(f"root {self.data_root}")
        print(f"split: {self.split}")
        print(f"mode: {'feature' if self.load_features else 'image' if self.load_images else '未知'}")
        print(f"num of classes: {len(self.class_dirs)}")
        print(f"num of samples: {len(self.samples)}")

def demo_usage():
    """
    数据加载器使用示例
    """
    # 数据集根目录
    data_root = '/home/stu12/homework/MLPR/data'
    
    # 创建数据加载器
    loader = DataLoader(data_root, split='train', mode='feature')

    # 获取单个样本
    features, label = loader[0]
    # print(features)
    print(f"特征形状: {features.shape}")
    print(f"特征类型: {type(features)}")
    print(f"标签: {label} ({loader.get_class_name(label)})")

    # # 批量获取特征
    # print("\n=== 批量特征加载 ===")
    # datas, labels = loader.get_data_and_labels()
    # print(f"所有特征形状: {datas.shape}")
    # print(f"所有标签形状: {labels.shape}")
    # print(f"标签分布: {np.bincount(labels)}")

    # # 统计datas中包含nan的数量：
    # nan_count = np.isnan(datas).sum()
    # print(f"特征数据中包含 NaN 的比例: {nan_count } / {datas.shape[0]} * {datas.shape[1]}")

if __name__ == "__main__":
    demo_usage()

