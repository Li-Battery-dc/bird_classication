class ViTConfig:
    """
    Vision Transformer配置类
    参数与ViT-Base/16类似
    适用于CUB-200数据集的微调任务
    """
    
    # ==================== 模型架构配置 ====================
    # 要符合ViT-Base/16架构，只能改正则参数或者分类头
    # 图像配置
    image_size = 224          # 输入图像大小
    patch_size = 16           # Patch大小 (16x16)
    in_channels = 3           # 输入通道数 (RGB)
    
    # Transformer配置
    embed_dim = 768           # Embedding维度 (ViT-Base)
    depth = 12                # Transformer Block层数
    num_heads = 12            # 注意力头数
    mlp_ratio = 4.0           # MLP隐藏层维度倍数 (768*4=3072)
    
    # Regularization
    dropout = 0.1             # Dropout比例
    attention_dropout = 0.1   # Attention dropout比例
    drop_path_rate = 0.2      # Stochastic Depth比例
    
    # 任务配置
    num_classes = 200         # CUB-200类别数
    
    # ==================== 预训练配置 ====================
    use_pretrained = True     # 是否使用预训练权重
    pretrained_model = 'vit_base_patch16_224_in21k'  # timm模型名称
    # 可选: 'vit_base_patch16_224', 'vit_base_patch16_224_in21k'

    # ==================== 训练配置 ====================

    data_root = './data/'
    result_dir = './result/vit/'

    batch_size_val = 256
     
    # ==================== Checkpoint恢复配置 ====================
    # 从checkpoint恢复训练，设置为checkpoint路径，如 'result/vit/train_xxx/ckpt/checkpoint_epoch_10.pth'
    resume_from_checkpoint = None
    
    # ==================== 阶段1: 只训练分类头 ====================
    stage1_epochs = 30
    stage1_batch_size = 384    
    stage1_warmup_epochs = 5
    stage1_base_lr = 1e-3     # LLRD 的base learning rate
    stage1_freeze_backbone = True  # 冻结backbone
    
    # ==================== 阶段2: 微调后几层 ====================
    stage2_epochs = 100          # 设为0则不启用该阶段
    stage2_batch_size = 256
    stage2_warmup_epochs = 15
    stage2_base_lr = 1e-4     # LLRD 的base learning rate
    stage2_unfreeze_layers = 4  # 解冻最后N个Transformer Block
    
    # ==================== 阶段3: 增加微调层数，统一学习率 ====================
    stage3_epochs = 150
    stage3_batch_size = 128
    stage3_warmup_epochs = 20
    stage3_base_lr = 3e-5   
    stage3_unfreeze_layers = 8  # 阶段3解冻最后N个Transformer Block
    
    # ==================== 优化器配置 ====================
    optimizer_type = 'adamw'  # 'adamw' 当前固定，改了这个参数也没用
    weight_decay = 0.05       # 权重衰减
    betas = (0.9, 0.999)      # AdamW betas
    
    # ==================== 学习率调度 ====================
    lr_scheduler = 'cosine'   # 当前使用自己实现的固定调度策略， 这个参数不起作用
    layer_decay = 0.75       # Layer-wise LR Decay系数 
    warmup_start_lr = 1e-6    # Warmup起始学习率
    min_lr = 1e-7             # 最小学习率

    # ==================== 损失函数配置 ====================
    # soft label时不起作用
    label_smoothing = 0.1     # 标签平滑 


    # 梯度裁剪
    clip_grad = True
    max_grad_norm = 1.0
    
    # ==================== 其他训练配置 ====================
    # 早停
    early_stopping = False    # 现在先全训练一段
    patience = 15             # 验证集N个epoch不提升则停止
    
    # 保存配置
    save_freq = 10            # 每N个epoch保存一次checkpoint
    save_best = True         # 保存最佳模型
    
    # 日志配置
    log_freq = 10             # 每个batch打印一次
    
    # 随机种子
    seed = 42

    # ==================== 数据增强配置 ====================
    # 训练时增强
    mixup_params = {
        'mixup_alpha': 0.2,
        'cutmix_alpha': 0.0,
        'prob': 0.5,
        'switch_prob': 0.5
    }
    train_augmentation = {
        'random_resized_crop': 224,  # RandomResizedCrop to 224x224
        'horizontal_flip': True,
        'color_jitter': {
            'brightness': 0.4,
            'contrast': 0.4, 
            'saturation': 0.4,
        },
        'rand_augment': {
            'n': 2,
            'm': 5
        },
        'random_rotation': 20,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'random_erasing': {
            'p': 0.1,
            'scale': (0.02, 0.2),
            'ratio': (0.3, 3.3),
            'value': 0
        }
    }
    
    # 验证时增强
    val_augmentation = {
        'resize': 256,
        'center_crop': 224,
        'normalize': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }


