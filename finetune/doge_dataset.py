#!/usr/bin/env python3
"""
DOGE数据集类
用于加载和处理DOGE微调数据
"""

import torch
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from doge_config import config

class DOGEDataset(Dataset):
    """DOGE数据集类"""
    
    def __init__(self, data_path, mode='train'):
        """
        初始化数据集
        
        Args:
            data_path (str): 数据文件路径
            mode (str): 数据集模式 ('train', 'val', 'test')
        """
        self.data_path = data_path
        self.mode = mode
        
        # 加载数据
        print(f"正在加载{mode}数据集: {data_path}")
        with open(data_path, 'rb') as f:
            self.windows = pickle.load(f)
        
        print(f"加载了 {len(self.windows)} 个{mode}窗口")
        
        # 数据统计
        self._print_stats()
    
    def _print_stats(self):
        """打印数据统计信息"""
        if len(self.windows) > 0:
            sample_window = self.windows[0]
            hist_data = sample_window['hist_data']
            pred_data = sample_window['pred_data']
            
            print(f"数据统计:")
            print(f"  - 历史数据形状: {hist_data.shape}")
            print(f"  - 预测数据形状: {pred_data.shape}")
            print(f"  - 历史特征列: {list(hist_data.columns)}")
            print(f"  - 价格范围: {hist_data[config.price_cols].min().min():.4f} - {hist_data[config.price_cols].max().max():.4f}")
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        获取单个数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含历史数据、预测目标、时间特征等
        """
        window = self.windows[idx]
        
        # 提取历史数据和预测数据
        hist_data = window['hist_data']
        pred_data = window['pred_data']
        
        # 准备输入特征
        hist_features = hist_data[config.feature_cols].values.astype(np.float32)
        hist_timestamps = hist_data[config.time_cols].values.astype(np.float32)
        
        # 准备预测目标
        pred_features = pred_data[config.feature_cols].values.astype(np.float32)
        pred_timestamps = pred_data[config.time_cols].values.astype(np.float32)
        
        # 转换为tensor
        hist_features = torch.from_numpy(hist_features)
        hist_timestamps = torch.from_numpy(hist_timestamps)
        pred_features = torch.from_numpy(pred_features)
        pred_timestamps = torch.from_numpy(pred_timestamps)
        
        # 构建返回字典
        sample = {
            'hist_features': hist_features,      # 历史特征 [seq_len, n_features]
            'hist_timestamps': hist_timestamps,  # 历史时间特征 [seq_len, n_time_features]
            'pred_features': pred_features,      # 预测目标 [pred_len, n_features]
            'pred_timestamps': pred_timestamps,  # 预测时间特征 [pred_len, n_time_features]
            'window_idx': torch.tensor(idx, dtype=torch.long),  # 窗口索引
        }
        
        # 如果有统计信息，转换为tensor
        if window.get('stats') is not None:
            stats = window['stats']
            sample['stats'] = {
                'price_mean': torch.from_numpy(stats['price_mean'].values.astype(np.float32)),
                'price_std': torch.from_numpy(stats['price_std'].values.astype(np.float32)),
                'vol_mean': torch.from_numpy(stats['vol_mean'].values.astype(np.float32)),
                'vol_std': torch.from_numpy(stats['vol_std'].values.astype(np.float32))
            }
        
        return sample

class DOGEDataLoader:
    """DOGE数据加载器"""
    
    def __init__(self, config):
        """
        初始化数据加载器
        
        Args:
            config: 配置对象
        """
        self.config = config
        
        # 创建数据集
        self.train_dataset = DOGEDataset(config.train_data_path, 'train')
        self.val_dataset = DOGEDataset(config.val_data_path, 'val')
        self.test_dataset = DOGEDataset(config.test_data_path, 'test')
        
        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    def get_loaders(self):
        """获取所有数据加载器"""
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_datasets(self):
        """获取所有数据集"""
        return self.train_dataset, self.val_dataset, self.test_dataset

def test_dataset():
    """测试数据集功能"""
    print("=" * 50)
    print("测试DOGE数据集")
    print("=" * 50)
    
    # 创建数据集
    dataset = DOGEDataset(config.train_data_path, 'train')
    
    # 获取一个样本
    sample = dataset[0]
    
    print(f"样本键: {list(sample.keys())}")
    print(f"历史特征形状: {sample['hist_features'].shape}")
    print(f"历史时间特征形状: {sample['hist_timestamps'].shape}")
    print(f"预测特征形状: {sample['pred_features'].shape}")
    print(f"预测时间特征形状: {sample['pred_timestamps'].shape}")
    
    # 创建数据加载器
    dataloader = DOGEDataLoader(config)
    train_loader, val_loader, test_loader = dataloader.get_loaders()
    
    print(f"\n数据加载器信息:")
    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")
    print(f"测试集批次数: {len(test_loader)}")
    
    # 获取一个批次
    for batch in train_loader:
        print(f"\n批次信息:")
        print(f"批次大小: {batch['hist_features'].shape[0]}")
        print(f"历史特征: {batch['hist_features'].shape}")
        print(f"预测特征: {batch['pred_features'].shape}")
        break
    
    print("=" * 50)
    print("数据集测试完成！")
    print("=" * 50)

if __name__ == "__main__":
    test_dataset()
