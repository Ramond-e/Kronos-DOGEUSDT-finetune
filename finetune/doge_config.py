#!/usr/bin/env python3
"""
DOGE微调专用配置文件
"""

import os

class DOGEConfig:
    """DOGE微调配置类"""
    
    def __init__(self):
        # =================================================================
        # 数据路径配置
        # =================================================================
        self.data_dir = "doge_data"
        self.raw_dir = os.path.join(self.data_dir, "raw")
        self.processed_dir = os.path.join(self.data_dir, "processed")
        self.splits_dir = os.path.join(self.data_dir, "splits")
        
        # 数据文件路径
        self.raw_data_path = os.path.join(self.raw_dir, "dogeusdt_1h_all_klines.csv")
        self.processed_data_path = os.path.join(self.processed_dir, "doge_processed.csv")
        self.train_data_path = os.path.join(self.splits_dir, "train_windows.pkl")
        self.val_data_path = os.path.join(self.splits_dir, "val_windows.pkl")
        self.test_data_path = os.path.join(self.splits_dir, "test_windows.pkl")
        
        # =================================================================
        # 数据参数
        # =================================================================
        self.lookback_window = 400      # 历史窗口大小
        self.predict_window = 120        # 预测窗口大小
        self.max_context = 512          # 最大上下文长度（Kronos-small限制）
        
        # 特征配置
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_cols = ['volume', 'amount']
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
        self.feature_cols = self.price_cols + self.vol_cols
        
        # =================================================================
        # 模型配置
        # =================================================================
        # 预训练模型路径（Hugging Face Hub）
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-small"
        
        # 模型参数
        self.s1_bits = 8               # Kronos-small的s1_bits
        self.s2_bits = 8                # Kronos-small的s2_bits
        self.d_model = 512              # 模型维度
        self.n_heads = 8                # 注意力头数
        self.n_layers = 12              # Transformer层数
        self.ff_dim = 2048              # 前馈网络维度
        
        # =================================================================
        # 训练参数
        # =================================================================
        self.epochs = 20                # 总训练轮数
        self.batch_size = 32            # 批次大小（适合8GB显存）
        self.gradient_accumulation_steps = 2  # 梯度累积步数
        
        # 学习率配置
        self.tokenizer_learning_rate = 5e-6   # Tokenizer学习率 (降低)
        self.predictor_learning_rate = 2e-6   # Predictor学习率 (大幅降低)
        self.warmup_steps = 1000              # 学习率预热步数
        self.weight_decay = 0.01              # 权重衰减
        
        # 优化器配置
        self.optimizer = "AdamW"
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        
        # 正则化
        self.dropout = 0.1
        self.clip_grad_norm = 3.0  # 增加梯度裁剪阈值
        
        # =================================================================
        # 训练策略
        # =================================================================
        self.mixed_precision = True     # 使用混合精度训练
        self.early_stopping_patience = 5  # 早停耐心值
        self.save_best_only = True      # 只保存最佳模型
        
        # 两阶段训练
        self.tokenizer_epochs = 5       # Tokenizer训练轮数
        self.predictor_epochs = 15      # Predictor训练轮数
        
        # =================================================================
        # 保存路径
        # =================================================================
        self.save_dir = "doge_outputs"
        self.tokenizer_save_dir = os.path.join(self.save_dir, "tokenizer")
        self.predictor_save_dir = os.path.join(self.save_dir, "predictor")
        self.logs_dir = os.path.join(self.save_dir, "logs")
        
        # 确保目录存在
        for dir_path in [self.save_dir, self.tokenizer_save_dir, 
                        self.predictor_save_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # =================================================================
        # 设备配置
        # =================================================================
        self.device = "cuda:0"          # 训练设备
        self.num_workers = 4            # 数据加载器工作进程数
        
        # =================================================================
        # 推理参数
        # =================================================================
        self.inference_T = 1.0          # 推理温度
        self.inference_top_p = 0.9      # 推理top-p
        self.inference_top_k = 0        # 推理top-k
        self.inference_sample_count = 1 # 推理采样数
        
        # =================================================================
        # 其他配置
        # =================================================================
        self.seed = 42                  # 随机种子
        self.log_interval = 100         # 日志间隔
        self.save_interval = 1000       # 保存间隔
        
    def print_config(self):
        """打印配置信息"""
        print("=" * 50)
        print("DOGE微调配置")
        print("=" * 50)
        
        print(f"数据配置:")
        print(f"  - 历史窗口: {self.lookback_window}")
        print(f"  - 预测窗口: {self.predict_window}")
        print(f"  - 最大上下文: {self.max_context}")
        
        print(f"\n模型配置:")
        print(f"  - Tokenizer: {self.pretrained_tokenizer_path}")
        print(f"  - Predictor: {self.pretrained_predictor_path}")
        print(f"  - 模型维度: {self.d_model}")
        print(f"  - 层数: {self.n_layers}")
        
        print(f"\n训练配置:")
        print(f"  - 批次大小: {self.batch_size}")
        print(f"  - 学习率: {self.predictor_learning_rate}")
        print(f"  - 训练轮数: {self.epochs}")
        print(f"  - 设备: {self.device}")
        
        print(f"\n保存路径:")
        print(f"  - 输出目录: {self.save_dir}")
        print(f"  - Tokenizer: {self.tokenizer_save_dir}")
        print(f"  - Predictor: {self.predictor_save_dir}")
        
        print("=" * 50)

# 创建全局配置实例
config = DOGEConfig()

if __name__ == "__main__":
    config.print_config()
