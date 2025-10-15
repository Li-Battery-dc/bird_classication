from .Network import CNNNetwork
from .Loss import FocalLoss
import torch

def validate_model(data_loader, device, state_dict_path, num_classes=200):

    # 加载模型
    model = CNNNetwork(num_classes=num_classes)
    
    # 加载 checkpoint
    state_dict = torch.load(state_dict_path, map_location=device)
    
    # 判断是否为字典格式（包含 'model_state_dict' 键）
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        # 完整的 checkpoint 格式，提取 model_state_dict
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        # 简单的 state_dict 格式，直接加载
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    # 使用相同的FocalLoss
    # 进行验证时不做warmup，直接使用目标gamma
    criterion = FocalLoss(num_classes=num_classes, target_gamma=2.0, smoothing=0.1)
    criterion.gamma = criterion.target_gamma

    total_loss = 0.0
    correct = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for batch_images, batch_labels in data_loader.get_batch_iterator(batch_size=256, shuffle=False):
            
            if isinstance(batch_images, torch.Tensor):
                batch_images = batch_images.to(device)
            else:
                batch_images = torch.tensor(batch_images, dtype=torch.float32).to(device)
            
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

            # 前向传播
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item() * batch_images.size(0)

            # 获取预测结果
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = total_loss / len(data_loader.samples)
    accuracy = correct / len(data_loader.samples)
    return avg_loss, accuracy