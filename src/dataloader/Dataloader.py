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
        self.class_to_label = {cls_name: idx for idx, cls_name in enumerate(self.class_dirs)}
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
            sample: (data_path, label) 元组
            
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
                    image = self.transform(image) # 处理变换为torch Tensor
                else:
                    # 默认转换为numpy数组并统一大小
                    image = np.array(image)
                    image = np.resize(image, (224, 224, 3))

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

    def get_batch_iterator(self, batch_size: int = 32, shuffle: bool = True, drop_last: bool = False):
        """
        获取批次迭代器，类似于标准PyTorch DataLoader的行为
        每次迭代返回一个batch的数据
        
        使用yield返回迭代器
        """
        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]

            # 防止改变batch_size的时候出错
            if drop_last and len(batch_indices) < batch_size:
                continue
            
            batch_data = []
            batch_labels = []
            
            for idx in batch_indices:
                data, label = self.load_sample(self.samples[idx])
                batch_data.append(data)
                batch_labels.append(label)
            
            # 转换为合适的格式
            if len(batch_data) > 0:
                if isinstance(batch_data[0], torch.Tensor):
                    batch_data = torch.stack(batch_data)
                else:
                    batch_data = np.stack(batch_data)
                batch_labels = np.array(batch_labels)
                
            yield batch_data, batch_labels

    def get_data_and_labels(self, num_samples=None, target_labels=None, target_class_names=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量获取指定数据和标签
        param num_samples: 如果指定，则只加载该数量的样本（用于快速测试）
        param target_classes: 如果指定，则只加载这些类别的样本（类别索引列表）
        param target_class_names: 如果指定，则只加载这些类别的样本（类别名称列表）
        
        Returns:
            (data(image/feature), labels) 两个numpy数组
        """
        print("loading data and labels...")
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

        data_array = np.stack(data_list)
        labels_array = np.array(labels_list)

        print(f"Loaded {self.split} data and labels:")
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
    # loader = DataLoader(data_root, split='train', mode='feature')

    # # 获取单个样本
    # features, label = loader[0]
    # # print(features)
    # print(f"特征形状: {features.shape}")
    # print(f"特征类型: {type(features)}")
    # print(f"标签: {label} ({loader.get_class_name(label)})")

    import torchvision.transforms as transforms
    train_transform  = transforms.Compose([
            transforms.RandomResizedCrop(224), # 224 * 224
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    image_loader = DataLoader(data_root, split='train', mode='image', transform=train_transform)
    images, labels = image_loader.get_data_and_labels(num_samples=10)
    print(f"图像形状: {images.shape}")
    print(f"图像类型: {type(images)}")
    print(f"标签: {labels} ({[image_loader.get_class_name(label) for label in labels]})")

if __name__ == "__main__":
    demo_usage()

