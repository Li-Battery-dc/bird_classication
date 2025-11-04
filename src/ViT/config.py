class ViTConfig:
    """
    Vision Transformer配置类
    参数与ViT-Base/16类似
    适用于CUB-200数据集的微调任务
    """
    
    # ==================== 模型架构配置 ====================
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
    
    # 任务配置
    num_classes = 200         # CUB-200类别数
    
    # ==================== 预训练配置 ====================
    use_pretrained = True     # 是否使用预训练权重
    pretrained_model = 'vit_base_patch16_224_in21k'  # timm模型名称
    # 可选: 'vit_base_patch16_224', 'vit_base_patch16_224_in21k'

    # ==================== 训练配置 ====================
    # 数据配置
    data_root = '/home/stu12/homework/MLPR/data/'
    result_dir = '/home/stu12/homework/MLPR/result/vit/'
    
    # 批次配置
    batch_size = 32           # 3090 24GB可以支持32-64
    
    # ==================== 阶段1: 只训练分类头 ====================
    stage1_epochs = 0
    stage1_lr_head = 1e-3     # 分类头学习率
    stage1_freeze_backbone = True  # 冻结backbone
    
    # ==================== 阶段2: 微调后几层 ====================
    stage2_epochs = 0          # 设为0则不启用该阶段
    stage2_lr_head = 1e-3     # 分类头学习率
    stage2_lr_backbone = 1e-4 # Backbone学习率
    stage2_unfreeze_layers = 3  # 解冻最后N个Transformer Block
    
    # ==================== 阶段3: 全模型微调 ====================
    stage3_epochs = 0
    stage3_lr = 1e-5          # 全模型统一学习率
    stage3_enabled = False    # 默认不启用，容易过拟合
    
    # ==================== 优化器配置 ====================
    optimizer_type = 'adamw'  # 'adamw' 当前固定
    weight_decay = 0.01       # 权重衰减
    betas = (0.9, 0.999)      # AdamW betas
    
    # ==================== 学习率调度 ====================
    lr_scheduler = 'cosine'   # 当前使用自己实现的固定调度策略， 这个参数不起作用
    warmup_epochs = 5         # Warmup轮数
    min_lr = 1e-6             # 最小学习率

    # ==================== 损失函数配置 ====================
    label_smoothing = 0.1     # 标签平滑

    # 梯度裁剪
    clip_grad = True
    max_grad_norm = 1.0
    
    # ==================== 其他训练配置 ====================
    # 早停
    early_stopping = True
    patience = 15             # 验证集N个epoch不提升则停止
    
    # 保存配置
    save_freq = 10            # 每N个epoch保存一次checkpoint
    save_best = True          # 保存最佳模型
    
    # 日志配置
    log_freq = 1             # 每个batch打印一次
    use_tensorboard = False   # 是否使用tensorboard
    
    # 随机种子
    seed = 42
    
    # ==================== 数据增强配置 ====================
    # 训练时增强
    train_augmentation = {
        'random_resized_crop': True,
        'crop_scale': (0.8, 1.0),
        'horizontal_flip': True,
        'color_jitter': {
            'brightness': 0.4,
            'contrast': 0.4, 
            'saturation': 0.4,
            'hue': 0.1
        },
        'random_rotation': 15,
        'random_erasing': False,  # Random Erasing增强
    }
    
    # 验证时增强
    val_augmentation = {
        'resize': 256,
        'center_crop': 224,
    }


