import torch
from torch.optim import Optimizer
import math

# 设置权重衰减和维护momentum缓冲区来提高泛化性
# 随机梯度下降体现在batch的选择是随机的。
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=5e-4):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.data

                # 添加 weight decay 惩罚大参数值
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                # Momentum
                if 'momentum_buffer' not in self.state[param]:
                    self.state[param]['momentum_buffer'] = torch.zeros_like(param.data)
                buf = self.state[param]['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                # 更新参数
                param.data.add_(buf, alpha=-lr)

        return loss

# 余弦退火学习率调度器
class LR_Scheduler:
    def __init__(self, optimizer, warmup_epochs=None, total_epochs=1000, min_lr=1e-6):
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch):
        # Cosine Annealing: lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * epoch / total_epochs))
        for i, group in enumerate(self.optimizer.param_groups):
            initial_lr = self.initial_lrs[i]
            if epoch < self.warmup_epochs:
                # Warmup: 固定初始学习率
                new_lr = initial_lr
            else:
                 # Cosine Annealing: 从 warmup 结束后开始退火
                # 调整 epoch 偏移，使退火从 warmup_epochs 开始
                adjusted_epoch = epoch - self.warmup_epochs
                adjusted_total = self.total_epochs - self.warmup_epochs
                new_lr = self.min_lr + 0.5 * (initial_lr - self.min_lr) * (
                    1 + math.cos(math.pi * adjusted_epoch / adjusted_total)
                )
            group['lr'] = new_lr
