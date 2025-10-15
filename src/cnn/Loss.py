import torch
import torch.nn.functional as F
import math

# 添加focal_weight 和label smoothing改善过拟合
class FocalLoss:
    def __init__(self, num_classes, warmup_epoch=None, target_gamma=2.0, smoothing=0.1):
        self.num_classes = num_classes
        self.gamma = 0.0
        self.target_gamma = target_gamma
        self.smoothing = smoothing
        self.warmup_epoch = warmup_epoch
        # 用于平滑过渡到调整target_gamma的情况
        self.transition_start_epoch = None
        self.transition_epochs = 50  # 过渡期长度
        self.old_target_gamma = target_gamma

    def set_target_gamma(self, new_gamma, current_epoch, transition_epochs=50):
        """平滑过渡到新的target_gamma，避免跳变"""
        self.old_target_gamma = self.target_gamma
        self.target_gamma = new_gamma
        self.transition_start_epoch = current_epoch
        self.transition_epochs = transition_epochs

    # warmup阶段，解决初期下降过慢
    def gamma_schedule(self, current_epoch):
        # 计算当前应该用的target_gamma（处理过渡期）
        current_target = self.target_gamma
        if self.transition_start_epoch is not None:
            epochs_since_transition = current_epoch - self.transition_start_epoch
            if 0 <= epochs_since_transition < self.transition_epochs:
                # 线性插值过渡
                alpha = epochs_since_transition / self.transition_epochs
                current_target = self.old_target_gamma + alpha * (self.target_gamma - self.old_target_gamma)
        
        # 计算gamma
        if self.warmup_epoch is not None and current_epoch < self.warmup_epoch:
            progress = (current_epoch / self.warmup_epoch)
            rate = (math.exp(progress) - 1) / (math.exp(1) - 1)  # 指数增长
        else:
            rate = 1.0
        self.gamma = rate * current_target

    
    def __call__(self, logits, targets):
        '''call方法兼容函数式调用：loss = criterion(outputs, batch_labels)'''

        log_probs = F.log_softmax(logits, dim=-1)
        
        # 得到focal loss的权重
        # 更关注pt 较低的难分类样本
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            # 降低confidence提高泛化性
            confidence = 1.0 - self.smoothing
            # 将one-hot标签展开为降低了confidence的标签分布
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), confidence)

        # sum(-true_dist * log_probs) 是带标签平滑的交叉熵
        loss = focal_weight * torch.sum(-true_dist * log_probs, dim=-1)

        return loss.mean()
