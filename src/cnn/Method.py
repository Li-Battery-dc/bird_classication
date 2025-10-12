import os
import os.path as osp

from .Network import CNNNetwork
from dataloader.Dataloader import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms

from .train import train_model
from .validate import validate

def cnnMethod(data_root='/home/stu12/homework/MLPR/data/', weight_root='/home/stu12/homework/MLPR/weights/cnn/', num_classes=200):

    train_transform = transform = transforms.Compose([
            transforms.RandomResizedCrop(224), # 224 * 224
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    train_loader = DataLoader(data_root, split='train', mode='image', transform=train_transform)

    # val_transform = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406],
    #                             [0.229, 0.224, 0.225])
    #     ])

    # val_loader = DataLoader(data_root, split='val', mode='image', transform=val_transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNNetwork(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(osp.join(weight_root, 'test_1.pth')))
    print("training on ", device)
    train_model(model, train_loader, device, num_epochs=100)
    torch.save(model.state_dict(), osp.join(weight_root, 'test_2.pth'))


if __name__ == "__main__":
    cnnMethod()