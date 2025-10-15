import torch
import math

# 设置权重衰减和维护momentum缓冲区来提高泛化性
# 随机梯度下降体现在batch的选择是随机的。
class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=5e-4):
        
        # 都转为list
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        else:
            params = list(params)
        
        # 初始化参数组, 同torch.optim
        self.param_groups = [{
            'params': params,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay
        }]
        self.state = {}
        
    def zero_grad(self):
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad.zero_()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for param in group['params']:
                # 只更新需要梯度且有梯度的参数
                if not param.requires_grad or param.grad is None:
                    continue
                
                grad = param.grad.data

                # 添加 weight decay 惩罚大参数值
                if weight_decay != 0:
                    grad = grad.add(param.data, alpha=weight_decay)

                # Momentum
                param_id = id(param) # state key 用id
                if param_id not in self.state:
                    self.state[param_id] = {'momentum_buffer': torch.zeros_like(param.data)}
                # 保存在state中
                buf = self.state[param_id]['momentum_buffer']
                buf.mul_(momentum).add_(grad)
                
                # 更新参数
                param.data.add_(buf, alpha=-lr)

        return loss
    
    def update_param_groups(self, params):
        """
        更新参数组，用于支持冻结策略
        当部分层被冻结后，只保留需要训练的参数
        """
        if isinstance(params, torch.nn.Parameter):
            params = [params]
        else:
            params = list(params)
        
        # 保存当前的超参数
        lr = self.param_groups[0]['lr']
        momentum = self.param_groups[0]['momentum']
        weight_decay = self.param_groups[0]['weight_decay']
        
        # 更新参数组
        self.param_groups = [{
            'params': params,
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay
        }]
        
        # 清理不再使用的参数的状态
        current_param_ids = {id(p) for p in params}
        state_keys_to_remove = [k for k in self.state.keys() if k not in current_param_ids]
        for key in state_keys_to_remove:
            del self.state[key]

# 余弦退火学习率调度器
class LR_Scheduler:
    def __init__(self, optimizer, warmup_epochs=0, total_epochs=1000, min_lr=1e-6):
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
