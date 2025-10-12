import torch
import torch.nn as nn
from torchvision import transforms
import random
import datetime

def train_epoch_with_iterator(model, data_loader, criterion, optimizer, device, batch_size=256):
    """
    支持batch加载和打乱的train方法
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total_samples = 0

    # 使用批次迭代器
    for batch_images, batch_labels in data_loader.get_batch_iterator(batch_size=batch_size, shuffle=True):

        if isinstance(batch_images, torch.Tensor):
            batch_images = batch_images.to(device)
        else:
            batch_images = torch.tensor(batch_images, dtype=torch.float32).to(device)
        
        batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()
        total_samples += batch_images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

def train_model(model, data_loader, device, num_epochs=100, batch_size=256):
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
    
        train_loss, train_acc = train_epoch_with_iterator(model, data_loader, criterion, optimizer, device, batch_size)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n"
                  f"Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%\n")
