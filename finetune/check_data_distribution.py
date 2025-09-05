#!/usr/bin/env python3
"""
DOGE数据分布检查脚本
检查数据标准化效果
"""

import torch
import sys
import os
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import KronosTokenizer, Kronos
from doge_config import config
from doge_dataset import DOGEDataLoader

def check_data_distribution():
    """检查数据分布"""
    print("=" * 50)
    print("DOGE数据分布检查")
    print("=" * 50)
    
    # 创建数据加载器
    dataloader = DOGEDataLoader(config)
    train_loader, _, _ = dataloader.get_loaders()
    
    # 收集统计数据
    all_hist_features = []
    all_pred_features = []
    
    print("正在收集数据统计...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 10:  # 只检查前10个批次
            break
            
        hist_features = batch['hist_features'].numpy()
        pred_features = batch['pred_features'].numpy()
        
        all_hist_features.append(hist_features)
        all_pred_features.append(pred_features)
        
        if batch_idx < 3:
            print(f"批次 {batch_idx + 1}:")
            print(f"  历史特征 - 均值: {hist_features.mean():.4f}, 标准差: {hist_features.std():.4f}")
            print(f"  历史特征 - 最小值: {hist_features.min():.4f}, 最大值: {hist_features.max():.4f}")
            print(f"  预测特征 - 均值: {pred_features.mean():.4f}, 标准差: {pred_features.std():.4f}")
            print(f"  预测特征 - 最小值: {pred_features.min():.4f}, 最大值: {pred_features.max():.4f}")
    
    # 合并所有数据
    all_hist = np.concatenate(all_hist_features, axis=0)
    all_pred = np.concatenate(all_pred_features, axis=0)
    
    print(f"\n总体统计 (前10个批次):")
    print(f"历史特征:")
    print(f"  均值: {all_hist.mean():.4f}")
    print(f"  标准差: {all_hist.std():.4f}")
    print(f"  最小值: {all_hist.min():.4f}")
    print(f"  最大值: {all_hist.max():.4f}")
    print(f"  25%分位数: {np.percentile(all_hist, 25):.4f}")
    print(f"  75%分位数: {np.percentile(all_hist, 75):.4f}")
    
    print(f"\n预测特征:")
    print(f"  均值: {all_pred.mean():.4f}")
    print(f"  标准差: {all_pred.std():.4f}")
    print(f"  最小值: {all_pred.min():.4f}")
    print(f"  最大值: {all_pred.max():.4f}")
    print(f"  25%分位数: {np.percentile(all_pred, 25):.4f}")
    print(f"  75%分位数: {np.percentile(all_pred, 75):.4f}")
    
    # 检查是否有异常值
    hist_outliers = np.abs(all_hist) > 5
    pred_outliers = np.abs(all_pred) > 5
    
    print(f"\n异常值检查 (|x| > 5):")
    print(f"历史特征异常值比例: {hist_outliers.sum() / hist_outliers.size:.4f}")
    print(f"预测特征异常值比例: {pred_outliers.sum() / pred_outliers.size:.4f}")

def check_gradient_norms():
    """检查梯度范数"""
    print("\n" + "=" * 50)
    print("梯度范数检查")
    print("=" * 50)
    
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    model = Kronos.from_pretrained(config.pretrained_predictor_path)
    tokenizer.to(device)
    model.to(device)
    
    # 创建数据加载器
    dataloader = DOGEDataLoader(config)
    train_loader, _, _ = dataloader.get_loaders()
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.predictor_learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adam_beta1, config.adam_beta2)
    )
    
    print("检查梯度范数...")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 3:  # 只检查前3个批次
            break
            
        # 移动数据到设备
        hist_features = batch['hist_features'].to(device)
        hist_timestamps = batch['hist_timestamps'].to(device)
        pred_features = batch['pred_features'].to(device)
        pred_timestamps = batch['pred_timestamps'].to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        # 合并历史和预测数据
        combined_features = torch.cat([hist_features, pred_features], dim=1)
        combined_timestamps = torch.cat([hist_timestamps, pred_timestamps], dim=1)
        
        # Tokenize输入数据
        with torch.no_grad():
            token_seq_0, token_seq_1 = tokenizer.encode(combined_features, half=True)
        
        # 准备输入和目标
        token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
        token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
        
        # 前向传播
        logits = model(token_in[0], token_in[1], combined_timestamps[:, :-1, :])
        loss, s1_loss, s2_loss = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数
        total_norm = 0
        param_count = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        print(f"批次 {batch_idx + 1}:")
        print(f"  损失: {loss.item():.6f}")
        print(f"  梯度范数: {total_norm:.6f}")
        print(f"  参数数量: {param_count}")
        
        # 应用梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
        
        # 计算裁剪后的梯度范数
        total_norm_clipped = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_clipped += param_norm.item() ** 2
        
        total_norm_clipped = total_norm_clipped ** (1. / 2)
        print(f"  裁剪后梯度范数: {total_norm_clipped:.6f}")
        print()

def main():
    """主函数"""
    check_data_distribution()
    check_gradient_norms()

if __name__ == "__main__":
    main()
