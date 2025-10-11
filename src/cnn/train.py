import torch
import torch.nn as nn
from torchvision import transforms

def train(model, images, labels, criterion, optimizer, device, batch_size=256):

    model.train()
    N = images.shape[0]
    # 手动实现batch训练
    permutation = torch.randperm(N)  # 随机打乱

    total_loss = 0.0
    correct = 0

    for i in range(0, N, batch_size):
        indices = permutation[i:i+batch_size]
        batch_images = images[indices]
        batch_labels = labels[indices]

        optimizer.zero_grad()
        outputs = model(batch_images)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_labels).sum().item()

    avg_loss = total_loss / N
    accuracy = correct / N
    return avg_loss, accuracy

def train_model(model, data_loader, device, num_epochs=100):
    
    images, labels = data_loader.get_data_and_labels()
    images = torch.tensor(images, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, images, labels, criterion, optimizer, device)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%")
