import os
import os.path as osp

from .Network import CNNNetwork
from dataloader.Dataloader import DataLoader
import torch
import torch.nn as nn
from torchvision import transforms

from .train import train_model
from .validate import validate_model

def train(weight_save_path, ckpt_load_path=None, data_root='/home/stu12/homework/MLPR/data/', num_classes=200, num_epochs=1000, batch_size=512):

    # 防止过拟合
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224), # 224 * 224
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])

    train_loader = DataLoader(data_root, split='train', mode='image', transform=train_transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNNetwork(num_classes=num_classes).to(device)
    print("training on ", device)
    if ckpt_load_path is not None:
        print(f"Resuming training from checkpoint: {ckpt_load_path}")
        train_model(model, train_loader, device, ckpt_load_path=ckpt_load_path, num_epochs=num_epochs, batch_size=batch_size)
    else:
        train_model(model, train_loader, device, ckpt_load_path=None, num_epochs=num_epochs, batch_size=batch_size)
    torch.save(model.state_dict(), weight_save_path)

def validate(weight_path, data_root='/home/stu12/homework/MLPR/data/', num_classes=200):
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

    val_loader = DataLoader(data_root, split='val', mode='image', transform=val_transform)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_loss, val_acc = validate_model(val_loader, device, state_dict_path=weight_path, num_classes=num_classes)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")

def cnnMethod(mode, save_pth_filename,data_root='/home/stu12/homework/MLPR/data/', weight_root='/home/stu12/homework/MLPR/result/cnn/weights', num_classes=200):
    if mode == 'train':
        if not os.path.exists(weight_root):
            os.makedirs(weight_root)
        weight_save_path = osp.join(weight_root, save_pth_filename)
        train(weight_save_path, data_root, num_classes)
    elif mode == 'validate':
        weight_path = osp.join(weight_root, save_pth_filename)
        validate(weight_path, data_root, num_classes)
    else:
        raise ValueError("Mode should be 'train' or 'validate'")

if __name__ == "__main__":
    cnnMethod()