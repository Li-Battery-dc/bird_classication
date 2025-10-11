from .Network import CNNNetwork
from dataloader.Dataloader import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms

def validate(data_root='/home/stu12/homework/MLPR/data', model_path='cnn_model.pth', num_classes=200):

    # 加载模型
    model = CNNNetwork(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    