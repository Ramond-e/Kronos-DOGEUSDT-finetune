#!/usr/bin/env python3
"""
DOGE微调训练脚本
两阶段训练：tokenizer微调 + predictor微调
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import time
import json
from datetime import datetime
from tqdm import tqdm
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import KronosTokenizer, Kronos, KronosPredictor
from doge_config import config
from doge_dataset import DOGEDataLoader

class DOGETrainer:
    """DOGE训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # 创建数据加载器
        self.dataloader = DOGEDataLoader(config)
        self.train_loader, self.val_loader, self.test_loader = self.dataloader.get_loaders()
        
        # 初始化模型
        self.tokenizer = None
        self.model = None
        self.predictor = None
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 设置日志
        self.setup_logging()
        
        print(f"训练器初始化完成，设备: {self.device}")
    
    def setup_logging(self):
        """设置日志"""
        log_file = os.path.join(config.logs_dir, f"doge_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_pretrained_models(self):
        """加载预训练模型"""
        self.logger.info("正在加载预训练模型...")
        
        try:
            # 加载tokenizer
            self.tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
            self.tokenizer.to(self.device)
            self.logger.info(f"Tokenizer加载成功: {config.pretrained_tokenizer_path}")
            
            # 加载Kronos模型
            self.model = Kronos.from_pretrained(config.pretrained_predictor_path)
            self.model.to(self.device)
            self.logger.info(f"Kronos模型加载成功: {config.pretrained_predictor_path}")
            
            # 创建predictor
            self.predictor = KronosPredictor(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_context=config.max_context,
                clip=5
            )
            self.logger.info("Predictor创建成功")
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            raise
    
    def train_tokenizer(self):
        """训练tokenizer"""
        self.logger.info("=" * 50)
        self.logger.info("开始Tokenizer微调")
        self.logger.info("=" * 50)
        
        # 设置优化器
        optimizer = optim.AdamW(
            self.tokenizer.parameters(),
            lr=config.tokenizer_learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=len(self.train_loader) * config.tokenizer_epochs
        )
        
        # 训练循环
        for epoch in range(config.tokenizer_epochs):
            self.current_epoch = epoch + 1
            
            # 训练一个epoch
            train_loss = self.train_tokenizer_epoch(optimizer, scheduler)
            
            # 验证
            val_loss = self.validate_tokenizer()
            
            # 记录日志
            self.logger.info(f"Epoch {self.current_epoch}/{config.tokenizer_epochs}")
            self.logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_tokenizer()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= config.early_stopping_patience:
                self.logger.info("早停触发，停止tokenizer训练")
                break
        
        self.logger.info("Tokenizer微调完成")
    
    def train_tokenizer_epoch(self, optimizer, scheduler):
        """训练tokenizer一个epoch"""
        self.tokenizer.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Tokenizer Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            hist_features = batch['hist_features'].to(self.device)
            pred_features = batch['pred_features'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 编码历史数据
            hist_tokens = self.tokenizer.encode(hist_features)
            
            # 编码预测目标
            pred_tokens = self.tokenizer.encode(pred_features)
            
            # 计算重建损失
            hist_recon = self.tokenizer.decode(hist_tokens)
            pred_recon = self.tokenizer.decode(pred_tokens)
            
            loss = nn.MSELoss()(hist_recon, hist_features) + nn.MSELoss()(pred_recon, pred_features)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), config.clip_grad_norm)
            
            # 优化器步进
            optimizer.step()
            scheduler.step()
            
            # 累积损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}'
            })
            
            # 日志记录
            if batch_idx % config.log_interval == 0:
                self.logger.info(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def validate_tokenizer(self):
        """验证tokenizer"""
        self.tokenizer.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移动数据到设备
                hist_features = batch['hist_features'].to(self.device)
                pred_features = batch['pred_features'].to(self.device)
                
                # 编码和解码
                hist_tokens = self.tokenizer.encode(hist_features)
                pred_tokens = self.tokenizer.encode(pred_features)
                
                hist_recon = self.tokenizer.decode(hist_tokens)
                pred_recon = self.tokenizer.decode(pred_tokens)
                
                # 计算损失
                loss = nn.MSELoss()(hist_recon, hist_features) + nn.MSELoss()(pred_recon, pred_features)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train_predictor(self):
        """训练predictor"""
        self.logger.info("=" * 50)
        self.logger.info("开始Predictor微调")
        self.logger.info("=" * 50)
        
        # 设置优化器
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.predictor_learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=len(self.train_loader) * config.predictor_epochs
        )
        
        # 重置早停
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 训练循环
        for epoch in range(config.predictor_epochs):
            self.current_epoch = epoch + 1
            
            # 训练一个epoch
            train_loss = self.train_predictor_epoch(optimizer, scheduler)
            
            # 验证
            val_loss = self.validate_predictor()
            
            # 记录日志
            self.logger.info(f"Epoch {self.current_epoch}/{config.predictor_epochs}")
            self.logger.info(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_predictor()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 早停检查
            if self.patience_counter >= config.early_stopping_patience:
                self.logger.info("早停触发，停止predictor训练")
                break
        
        self.logger.info("Predictor微调完成")
    
    def train_predictor_epoch(self, optimizer, scheduler):
        """训练predictor一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Predictor Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            hist_features = batch['hist_features'].to(self.device)
            hist_timestamps = batch['hist_timestamps'].to(self.device)
            pred_features = batch['pred_features'].to(self.device)
            pred_timestamps = batch['pred_timestamps'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 合并历史和预测数据
            combined_features = torch.cat([hist_features, pred_features], dim=1)  # [batch, seq_len+pred_len, features]
            combined_timestamps = torch.cat([hist_timestamps, pred_timestamps], dim=1)  # [batch, seq_len+pred_len, time_features]
            
            # Tokenize输入数据
            with torch.no_grad():
                token_seq_0, token_seq_1 = self.tokenizer.encode(combined_features, half=True)
            
            # 准备输入和目标
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            
            # 前向传播
            logits = self.model(token_in[0], token_in[1], combined_timestamps[:, :-1, :])
            loss, s1_loss, s2_loss = self.model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.clip_grad_norm)
            
            # 优化器步进
            optimizer.step()
            scheduler.step()
            
            # 累积损失
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/num_batches:.6f}'
            })
            
            # 日志记录
            if batch_idx % config.log_interval == 0:
                self.logger.info(f"Batch {batch_idx}: Loss = {loss.item():.6f}")
        
        return total_loss / num_batches
    
    def validate_predictor(self):
        """验证predictor"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移动数据到设备
                hist_features = batch['hist_features'].to(self.device)
                hist_timestamps = batch['hist_timestamps'].to(self.device)
                pred_features = batch['pred_features'].to(self.device)
                pred_timestamps = batch['pred_timestamps'].to(self.device)
                
                # 合并历史和预测数据
                combined_features = torch.cat([hist_features, pred_features], dim=1)
                combined_timestamps = torch.cat([hist_timestamps, pred_timestamps], dim=1)
                
                # Tokenize输入数据
                token_seq_0, token_seq_1 = self.tokenizer.encode(combined_features, half=True)
                
                # 准备输入和目标
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                
                # 前向传播
                logits = self.model(token_in[0], token_in[1], combined_timestamps[:, :-1, :])
                loss, _, _ = self.model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_tokenizer(self):
        """保存tokenizer"""
        save_path = os.path.join(config.tokenizer_save_dir, f"best_tokenizer_epoch_{self.current_epoch}.pt")
        torch.save(self.tokenizer.state_dict(), save_path)
        self.logger.info(f"Tokenizer已保存: {save_path}")
    
    def save_predictor(self):
        """保存predictor"""
        save_path = os.path.join(config.predictor_save_dir, f"best_predictor_epoch_{self.current_epoch}.pt")
        torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Predictor已保存: {save_path}")
    
    def run_training(self):
        """运行完整训练流程"""
        self.logger.info("=" * 50)
        self.logger.info("开始DOGE微调训练")
        self.logger.info("=" * 50)
        
        # 加载预训练模型
        self.load_pretrained_models()
        
        # 阶段1: Tokenizer微调
        self.train_tokenizer()
        
        # 阶段2: Predictor微调
        self.train_predictor()
        
        self.logger.info("=" * 50)
        self.logger.info("DOGE微调训练完成！")
        self.logger.info("=" * 50)

def main():
    """主函数"""
    # 检查GPU
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("警告: GPU不可用，将使用CPU训练")
    
    # 创建训练器
    trainer = DOGETrainer()
    
    # 开始训练
    start_time = time.time()
    trainer.run_training()
    end_time = time.time()
    
    # 打印训练时间
    training_time = end_time - start_time
    print(f"总训练时间: {training_time/3600:.2f} 小时")

if __name__ == "__main__":
    main()
