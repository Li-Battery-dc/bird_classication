import torch
import random
import datetime
import os

from .Optimizer import SGD, LR_Scheduler
from .Loss import FocalLoss

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

def load_checkpoint(checkpoint_path, model, optimizer):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    return model, optimizer, start_epoch

def train_model(model, data_loader, device, 
                ckpt_load_path=None, result_dir="/home/stu12/homework/MLPR/result/cnn/", 
                enable_freeze=True, freeze_epoch_ratio=0.7,
                enable_warmup=True, warmup_ratio=0.2,
                num_epochs=1000, batch_size=256):
    """
    训练模型,可选使用渐进式冻结策略，以及warmup策略
    
    """

    warmup_epochs = int(num_epochs * warmup_ratio) if enable_warmup else None
    freeze_start_epoch = int(num_epochs * freeze_epoch_ratio) if enable_freeze else num_epochs + 1
    
    criterion = FocalLoss(num_classes=200, warmup_epoch=warmup_epochs, target_gamma=2.0, smoothing=0.1)
    optimizer = SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = LR_Scheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=num_epochs, min_lr=1e-6)

    if ckpt_load_path is not None and os.path.exists(ckpt_load_path):
        model, optimizer, start_epoch = load_checkpoint(ckpt_load_path, model, optimizer)
        print(f"Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        
    log_dir = os.path.join(result_dir, "logs")
    ckpt_dir = os.path.join(result_dir, "ckpts", f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_file = os.path.join(log_dir, f"train_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(log_file, "w") as log:
        log.write("Training log at {}\n".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        log.write(f"Freeze strategy: {'Enabled' if enable_freeze else 'Disabled'}\n")
        if enable_freeze:
            log.write(f"Freeze starts at epoch {freeze_start_epoch}\n")
        log.write("-" * 80 + "\n")
        
        for epoch in range(start_epoch, num_epochs):
            # 在指定epoch开始进入冻结深训练
            if enable_freeze and epoch == freeze_start_epoch:
                # 冻结模型层
                model.freeze_partial()
                trainable_param_list = [p for p in model.parameters() if p.requires_grad]
                optimizer.update_param_groups(trainable_param_list)
                # 增强FocalLoss的gamma,更关注难样本
                criterion.target_gamma = 3.0
            
            train_loss, train_acc = train_epoch_with_iterator(model, data_loader, criterion, optimizer, device, batch_size)
            lr_scheduler.step(epoch)
            criterion.gamma_schedule(epoch)
            
            current_lr = optimizer.param_groups[0]['lr']
            freeze_status = "partial frozen" if enable_freeze and epoch >= freeze_start_epoch else "Full"

            log_message = (f"Epoch [{epoch+1}/{num_epochs}] [{freeze_status}], time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n"
                           f"Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}%, Learning Rate: {current_lr:.6f}\n")

            # 打印到控制台
            print(log_message)
            # 写入日志文件
            log.write(log_message)
            
            if (epoch + 1) % 50 == 0:
                # 保存ckpts
                checkpoint_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'accuracy': train_acc
                }, checkpoint_path)