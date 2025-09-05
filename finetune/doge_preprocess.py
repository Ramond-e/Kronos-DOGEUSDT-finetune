#!/usr/bin/env python3
"""
DOGE数据预处理脚本
将DOGE CSV数据转换为Kronos训练格式
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DOGEDataPreprocessor:
    """DOGE数据预处理器"""
    
    def __init__(self, data_dir="doge_data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.splits_dir = os.path.join(data_dir, "splits")
        
        # 确保目录存在
        for dir_path in [self.raw_dir, self.processed_dir, self.splits_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_raw_data(self):
        """加载原始DOGE数据"""
        print("正在加载原始DOGE数据...")
        
        # 加载CSV数据
        csv_path = os.path.join(self.raw_dir, "dogeusdt_1h_all_klines.csv")
        df = pd.read_csv(csv_path)
        
        print(f"原始数据形状: {df.shape}")
        print(f"数据列: {list(df.columns)}")
        
        return df
    
    def preprocess_data(self, df):
        """预处理数据"""
        print("正在预处理数据...")
        
        # 转换时间戳
        df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')
        
        # 选择需要的列
        required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
        df = df[required_cols].copy()
        
        # 添加amount列（用volume * 平均价格估算）
        df['amount'] = df['volume'] * df[['open', 'high', 'low', 'close']].mean(axis=1)
        
        # 按时间排序
        df = df.sort_values('timestamps').reset_index(drop=True)
        
        # 检查数据完整性
        print(f"数据时间范围: {df['timestamps'].min()} 到 {df['timestamps'].max()}")
        print(f"数据条数: {len(df)}")
        print(f"缺失值检查:")
        print(df.isnull().sum())
        
        return df
    
    def generate_time_features(self, df):
        """生成时间特征"""
        print("正在生成时间特征...")
        
        # 基础时间特征
        df['minute'] = df['timestamps'].dt.minute
        df['hour'] = df['timestamps'].dt.hour
        df['weekday'] = df['timestamps'].dt.weekday
        df['day'] = df['timestamps'].dt.day
        df['month'] = df['timestamps'].dt.month
        
        return df
    
    def create_sliding_windows(self, df, lookback=400, predict=120):
        """创建滑动窗口数据"""
        print(f"正在创建滑动窗口 (lookback={lookback}, predict={predict})...")
        
        windows = []
        total_len = len(df)
        
        for i in range(lookback, total_len - predict + 1):
            # 历史数据窗口
            hist_data = df.iloc[i-lookback:i]
            # 预测目标窗口
            pred_data = df.iloc[i:i+predict]
            
            window = {
                'hist_data': hist_data.copy(),
                'pred_data': pred_data.copy(),
                'start_idx': i-lookback,
                'end_idx': i+predict-1
            }
            windows.append(window)
        
        print(f"创建了 {len(windows)} 个滑动窗口")
        return windows
    
    def split_data(self, windows, train_ratio=0.7, val_ratio=0.15):
        """分割数据为训练/验证/测试集"""
        print("正在分割数据集...")
        
        total_windows = len(windows)
        train_end = int(total_windows * train_ratio)
        val_end = int(total_windows * (train_ratio + val_ratio))
        
        train_windows = windows[:train_end]
        val_windows = windows[train_end:val_end]
        test_windows = windows[val_end:]
        
        print(f"训练集: {len(train_windows)} 个窗口")
        print(f"验证集: {len(val_windows)} 个窗口")
        print(f"测试集: {len(test_windows)} 个窗口")
        
        return train_windows, val_windows, test_windows
    
    def normalize_data(self, windows, method='zscore'):
        """标准化数据"""
        print(f"正在标准化数据 (方法: {method})...")
        
        # 收集所有历史数据的统计信息
        all_hist_data = []
        for window in windows:
            all_hist_data.append(window['hist_data'])
        
        combined_hist = pd.concat(all_hist_data)
        
        # 计算统计量 - 使用更稳健的方法
        price_cols = ['open', 'high', 'low', 'close']
        vol_cols = ['volume', 'amount']
        
        # 使用中位数和MAD来减少异常值影响
        price_mean = combined_hist[price_cols].median()
        # 手动计算MAD
        price_mad = combined_hist[price_cols].sub(price_mean).abs().median()
        price_std = price_mad * 1.4826  # MAD到标准差的转换
        
        vol_mean = combined_hist[vol_cols].median()
        vol_mad = combined_hist[vol_cols].sub(vol_mean).abs().median()
        vol_std = vol_mad * 1.4826
        
        # 确保标准差不为0
        price_std = price_std.clip(lower=1e-8)
        vol_std = vol_std.clip(lower=1e-8)
        
        print(f"标准化统计 (使用中位数和MAD):")
        print(f"  价格均值: {price_mean.values}")
        print(f"  价格标准差: {price_std.values}")
        print(f"  成交量均值: {vol_mean.values}")
        print(f"  成交量标准差: {vol_std.values}")
        
        # 标准化每个窗口
        normalized_windows = []
        for window in windows:
            norm_window = window.copy()
            
            # 标准化历史数据
            hist_data = window['hist_data'].copy()
            hist_data[price_cols] = (hist_data[price_cols] - price_mean) / price_std
            hist_data[vol_cols] = (hist_data[vol_cols] - vol_mean) / vol_std
            
            # 限制异常值到[-5, 5]范围
            hist_data[price_cols] = hist_data[price_cols].clip(-5, 5)
            hist_data[vol_cols] = hist_data[vol_cols].clip(-5, 5)
            norm_window['hist_data'] = hist_data
            
            # 标准化预测数据
            pred_data = window['pred_data'].copy()
            pred_data[price_cols] = (pred_data[price_cols] - price_mean) / price_std
            pred_data[vol_cols] = (pred_data[vol_cols] - vol_mean) / vol_std
            
            # 限制异常值到[-5, 5]范围
            pred_data[price_cols] = pred_data[price_cols].clip(-5, 5)
            pred_data[vol_cols] = pred_data[vol_cols].clip(-5, 5)
            norm_window['pred_data'] = pred_data
            
            # 保存统计信息用于反标准化
            norm_window['stats'] = {
                'price_mean': price_mean,
                'price_std': price_std,
                'vol_mean': vol_mean,
                'vol_std': vol_std
            }
            
            normalized_windows.append(norm_window)
        
        return normalized_windows
    
    def save_processed_data(self, train_windows, val_windows, test_windows):
        """保存处理后的数据"""
        print("正在保存处理后的数据...")
        
        # 保存训练集
        train_path = os.path.join(self.splits_dir, "train_windows.pkl")
        with open(train_path, 'wb') as f:
            pickle.dump(train_windows, f)
        
        # 保存验证集
        val_path = os.path.join(self.splits_dir, "val_windows.pkl")
        with open(val_path, 'wb') as f:
            pickle.dump(val_windows, f)
        
        # 保存测试集
        test_path = os.path.join(self.splits_dir, "test_windows.pkl")
        with open(test_path, 'wb') as f:
            pickle.dump(test_windows, f)
        
        print(f"数据已保存到: {self.splits_dir}")
        print(f"- 训练集: {train_path}")
        print(f"- 验证集: {val_path}")
        print(f"- 测试集: {test_path}")
    
    def run_preprocessing(self, lookback=400, predict=120):
        """运行完整的预处理流程"""
        print("=" * 50)
        print("开始DOGE数据预处理")
        print("=" * 50)
        
        # 1. 加载原始数据
        df = self.load_raw_data()
        
        # 2. 预处理数据
        df = self.preprocess_data(df)
        
        # 3. 生成时间特征
        df = self.generate_time_features(df)
        
        # 4. 保存处理后的完整数据
        processed_path = os.path.join(self.processed_dir, "doge_processed.csv")
        df.to_csv(processed_path, index=False)
        print(f"处理后的完整数据已保存: {processed_path}")
        
        # 5. 创建滑动窗口
        windows = self.create_sliding_windows(df, lookback, predict)
        
        # 6. 分割数据
        train_windows, val_windows, test_windows = self.split_data(windows)
        
        # 7. 标准化数据
        train_windows = self.normalize_data(train_windows)
        val_windows = self.normalize_data(val_windows)
        test_windows = self.normalize_data(test_windows)
        
        # 8. 保存数据
        self.save_processed_data(train_windows, val_windows, test_windows)
        
        print("=" * 50)
        print("数据预处理完成！")
        print("=" * 50)
        
        return train_windows, val_windows, test_windows

def main():
    """主函数"""
    preprocessor = DOGEDataPreprocessor()
    train_windows, val_windows, test_windows = preprocessor.run_preprocessing()
    
    print(f"\n预处理结果总结:")
    print(f"- 训练窗口数: {len(train_windows)}")
    print(f"- 验证窗口数: {len(val_windows)}")
    print(f"- 测试窗口数: {len(test_windows)}")
    print(f"- 每个窗口: {len(train_windows[0]['hist_data'])} 历史 + {len(train_windows[0]['pred_data'])} 预测")

if __name__ == "__main__":
    main()
