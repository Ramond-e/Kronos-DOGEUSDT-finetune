#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kronos DOGE/USDT 模型预测服务
基于kronos_prediction_eval.py的高精度预测方法
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
sys.path.append('../')

# 导入Kronos模型
try:
    from model.kronos import KronosTokenizer, Kronos, KronosPredictor
    KRONOS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Kronos模型导入失败: {e}")
    KRONOS_AVAILABLE = False


def convert_to_beijing_time(timestamp_series, format_str='%Y-%m-%d %H:%M:%S'):
    """
    将时间戳转换为北京时间显示格式
    
    Args:
        timestamp_series: pandas Series或单个时间戳
        format_str: 时间格式字符串
        
    Returns:
        转换后的北京时间字符串或列表
    """
    beijing_tz = pytz.timezone('Asia/Shanghai')
    
    if isinstance(timestamp_series, pd.Series):
        # 如果是UTC时间，先转换为UTC时区感知，再转换为北京时间
        if timestamp_series.dt.tz is None:
            # 假设原始时间是UTC时间
            utc_times = timestamp_series.dt.tz_localize('UTC')
        else:
            utc_times = timestamp_series.dt.tz_convert('UTC')
        
        # 转换为北京时间
        beijing_times = utc_times.dt.tz_convert(beijing_tz)
        return beijing_times.dt.strftime(format_str).tolist()
    else:
        # 单个时间戳
        if isinstance(timestamp_series, str):
            timestamp_series = pd.to_datetime(timestamp_series)
        
        if timestamp_series.tz is None:
            # 假设是UTC时间
            utc_time = timestamp_series.tz_localize('UTC')
        else:
            utc_time = timestamp_series.tz_convert('UTC')
        
        # 转换为北京时间
        beijing_time = utc_time.tz_convert(beijing_tz)
        return beijing_time.strftime(format_str)


class KronosModelService:
    """Kronos模型预测服务类"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.predictor = None
        self.device = None
        self.loaded = False
        self.model_info = {}
    
    def check_availability(self):
        """检查模型是否可用"""
        return KRONOS_AVAILABLE
    
    def load_model(self, device=None):
        """
        加载Kronos模型
        
        Args:
            device: 设备选择 ('cpu', 'cuda', 'mps', 'auto')
        """
        if not KRONOS_AVAILABLE:
            raise RuntimeError("Kronos模型不可用，请检查model模块安装")
        
        if self.loaded:
            print("✅ 模型已加载")
            return True
        
        try:
            print("🔄 开始加载Kronos DOGE模型...")
            
            # 设备选择
            self.device = self._select_device(device)
            print(f"🖥️ 选择设备: {self.device}")
            
            # 从Hugging Face加载微调后的模型
            print("📦 从Hugging Face加载微调模型...")
            self.tokenizer = KronosTokenizer.from_pretrained("Ramond-e/doge-kronos-tokenizer")
            self.model = Kronos.from_pretrained("Ramond-e/doge-kronos-predictor")
            
            # 创建预测器 - 使用评估文件中的最佳配置
            print("🔧 创建预测器...")
            self.predictor = KronosPredictor(
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_context=512,  # 与评估文件一致
                clip=5           # 数据标准化剪切范围
            )
            
            # 记录模型信息
            self.model_info = {
                'tokenizer_source': 'Ramond-e/doge-kronos-tokenizer',
                'model_source': 'Ramond-e/doge-kronos-predictor',
                'device': self.device,
                'max_context': 512,
                'clip_range': 5,
                'loaded_at': datetime.now().isoformat()
            }
            
            self.loaded = True
            print("✅ Kronos模型加载成功!")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.loaded = False
            return False
    
    def _select_device(self, device):
        """智能设备选择"""
        if device == 'auto' or device is None:
            # 自动选择最佳设备
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        
        elif device == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                print("⚠️ CUDA不可用，回退到CPU")
                return 'cpu'
        
        elif device == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                print("⚠️ MPS不可用，回退到CPU")
                return 'cpu'
        
        else:
            return 'cpu'
    
    def predict_future(self, data_df, pred_hours=120, temperature=1.0, top_p=0.9, sample_count=1):
        """
        预测未来价格趋势
        
        Args:
            data_df: 历史数据DataFrame (必须包含OHLCV和amount列)
            pred_hours: 预测小时数 (默认120小时，5天)
            temperature: 温度参数，控制预测随机性 (0.1-2.0)
            top_p: 核采样参数，控制预测多样性 (0.1-1.0)
            sample_count: 采样次数 (1-5)
        
        Returns:
            dict: 包含预测结果和分析的字典
        """
        if not self.loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")
        
        # 参数验证
        self._validate_prediction_params(temperature, top_p, sample_count)
        
        # 数据验证和准备
        prepared_data = self._prepare_data(data_df)
        
        try:
            print(f"🔮 开始预测: T={temperature}, top_p={top_p}, samples={sample_count}")
            print(f"📊 历史数据: {len(prepared_data['x_df'])}小时, 预测长度: {pred_hours}小时")
            
            # 执行预测 - 使用与评估文件相同的方法
            pred_df = self.predictor.predict(
                df=prepared_data['x_df'],
                x_timestamp=prepared_data['x_timestamp'],
                y_timestamp=prepared_data['y_timestamp'],
                pred_len=pred_hours,
                T=temperature,
                top_k=0,         # 关闭top-k采样，使用top-p
                top_p=top_p,
                sample_count=sample_count,
                verbose=True     # 显示预测进度
            )
            
            print("✅ 预测完成!")
            
            # 后处理和分析
            result = self._post_process_prediction(
                historical_df=prepared_data['x_df'],
                prediction_df=pred_df,
                historical_timestamps=prepared_data['x_timestamp'],
                prediction_timestamps=prepared_data['y_timestamp'],
                parameters={
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'pred_hours': pred_hours,
                    'historical_hours': len(prepared_data['x_df'])
                }
            )
            
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            raise RuntimeError(f"预测执行失败: {str(e)}")
    
    def _validate_prediction_params(self, temperature, top_p, sample_count):
        """验证预测参数"""
        if not (0.1 <= temperature <= 2.0):
            raise ValueError("Temperature必须在0.1-2.0之间")
        if not (0.1 <= top_p <= 1.0):
            raise ValueError("top_p必须在0.1-1.0之间")
        if not (1 <= sample_count <= 5):
            raise ValueError("sample_count必须在1-5之间")
    
    def _prepare_data(self, data_df):
        """准备预测数据"""
        # 验证必需列
        required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        if missing_cols:
            raise ValueError(f"数据缺少必需列: {missing_cols}")
        
        # 数据清理
        df = data_df.copy()
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        df = df.sort_values('timestamps').reset_index(drop=True)
        
        # 检查数据质量
        if df[['open', 'high', 'low', 'close', 'volume', 'amount']].isnull().any().any():
            raise ValueError("数据包含空值，请检查数据质量")
        
        # 使用最新400小时数据作为历史数据（与评估文件一致）
        lookback = min(400, len(df))
        x_df = df.tail(lookback)[['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = df.tail(lookback)['timestamps'].copy()
        
        # 生成未来时间戳
        last_time = x_timestamp.iloc[-1]
        future_timestamps = pd.date_range(
            start=last_time + pd.Timedelta(hours=1),
            periods=120,  # 固定预测120小时
            freq='h'
        )
        y_timestamp = pd.Series(future_timestamps)
        
        return {
            'x_df': x_df.reset_index(drop=True),
            'x_timestamp': x_timestamp.reset_index(drop=True),
            'y_timestamp': y_timestamp,
            'last_price': df['close'].iloc[-1],
            'last_volume': df['volume'].iloc[-1]
        }
    
    def _post_process_prediction(self, historical_df, prediction_df, historical_timestamps, 
                                prediction_timestamps, parameters):
        """后处理预测结果"""
        
        # 添加时间戳
        prediction_df = prediction_df.copy()
        prediction_df['timestamps'] = prediction_timestamps.values
        
        # 基础统计
        current_price = historical_df['close'].iloc[-1]
        future_price = prediction_df['close'].iloc[-1]
        price_change = future_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # 趋势分析
        trend_analysis = self._analyze_trend(historical_df, prediction_df, current_price)
        
        # 波动性分析
        volatility_analysis = self._analyze_volatility(historical_df, prediction_df)
        
        # 图表数据
        chart_data = self._generate_chart_data(historical_df, prediction_df, 
                                              historical_timestamps, prediction_timestamps)
        
        # 预测质量指标
        quality_metrics = self._calculate_quality_metrics(prediction_df, parameters)
        
        return {
            'success': True,
            'prediction_data': prediction_df.to_dict('records'),
            'historical_data': historical_df.to_dict('records'),
            'trend_analysis': trend_analysis,
            'volatility_analysis': volatility_analysis,
            'chart_data': chart_data,
            'quality_metrics': quality_metrics,
            'parameters': parameters,
            'model_info': self.model_info,
            'generated_at': datetime.now().isoformat()
        }
    
    def _analyze_trend(self, historical_df, prediction_df, current_price):
        """趋势分析"""
        future_price = prediction_df['close'].iloc[-1]
        price_change = future_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # 趋势强度分类
        if price_change_pct > 5:
            trend = "强烈上涨"
            trend_color = "#00C851"
            trend_emoji = "🚀"
        elif price_change_pct > 2:
            trend = "上涨"
            trend_color = "#4CAF50"
            trend_emoji = "📈"
        elif price_change_pct > -2:
            trend = "横盘"
            trend_color = "#FF9800"
            trend_emoji = "➡️"
        elif price_change_pct > -5:
            trend = "下跌"
            trend_color = "#FF5722"
            trend_emoji = "📉"
        else:
            trend = "强烈下跌"
            trend_color = "#F44336"
            trend_emoji = "⬇️"
        
        # 价格区间
        pred_min = prediction_df['close'].min()
        pred_max = prediction_df['close'].max()
        
        # 历史波动率
        hist_returns = historical_df['close'].pct_change().dropna()
        hist_volatility = hist_returns.std() * np.sqrt(24) * 100  # 年化波动率
        
        return {
            'current_price': float(current_price),
            'predicted_price': float(future_price),
            'price_change': float(price_change),
            'price_change_pct': float(price_change_pct),
            'trend': trend,
            'trend_color': trend_color,
            'trend_emoji': trend_emoji,
            'historical_volatility': float(hist_volatility),
            'predicted_range': {
                'min': float(pred_min),
                'max': float(pred_max),
                'range_pct': float((pred_max - pred_min) / current_price * 100)
            }
        }
    
    def _analyze_volatility(self, historical_df, prediction_df):
        """波动性分析"""
        # 历史波动率
        hist_returns = historical_df['close'].pct_change().dropna()
        hist_volatility = hist_returns.std() * np.sqrt(24) * 100  # 年化波动率
        
        # 预测波动率
        pred_returns = prediction_df['close'].pct_change().dropna()
        pred_volatility = pred_returns.std() * np.sqrt(24) * 100  # 年化波动率
        
        return {
            'historical_volatility': float(hist_volatility),
            'predicted_volatility': float(pred_volatility),
            'volatility_change': float(pred_volatility - hist_volatility),
            'volatility_level': self._classify_volatility(pred_volatility)
        }
    
    def _classify_volatility(self, volatility):
        """波动率水平分类"""
        if volatility < 20:
            return "低波动"
        elif volatility < 40:
            return "中等波动"
        elif volatility < 60:
            return "高波动"
        else:
            return "极高波动"
    
    def _generate_chart_data(self, historical_df, prediction_df, hist_timestamps, pred_timestamps):
        """生成图表数据（时间显示为北京时间）"""
        return {
            'historical': {
                'timestamps': convert_to_beijing_time(hist_timestamps),
                'open': historical_df['open'].tolist(),
                'high': historical_df['high'].tolist(),
                'low': historical_df['low'].tolist(),
                'close': historical_df['close'].tolist(),
                'volume': historical_df['volume'].tolist()
            },
            'prediction': {
                'timestamps': convert_to_beijing_time(pred_timestamps),
                'open': prediction_df['open'].tolist(),
                'high': prediction_df['high'].tolist(),
                'low': prediction_df['low'].tolist(),
                'close': prediction_df['close'].tolist(),
                'volume': prediction_df['volume'].tolist()
            }
        }
    
    def _calculate_quality_metrics(self, prediction_df, parameters):
        """计算预测质量指标"""
        # 基于温度和top_p计算预测质量评分
        temp_score = 100 - abs(parameters['temperature'] - 1.0) * 30  # 最佳温度1.0
        diversity_score = parameters['top_p'] * 100  # top_p越高多样性越好
        sample_score = min(parameters['sample_count'] * 20, 100)  # 样本数加分
        
        overall_score = (temp_score + diversity_score + sample_score) / 3
        
        # 预测稳定性（基于价格变化的标准差）
        price_stability = 100 - (prediction_df['close'].std() / prediction_df['close'].mean() * 100)
        price_stability = max(0, min(100, price_stability))
        
        return {
            'overall_quality_score': float(overall_score),
            'temperature_optimality': float(temp_score),
            'diversity_score': float(diversity_score),
            'sample_adequacy': float(sample_score),
            'prediction_stability': float(price_stability),
            'confidence_level': self._calculate_confidence(overall_score)
        }
    
    def _calculate_confidence(self, quality_score):
        """计算置信度水平"""
        if quality_score >= 90:
            return "非常高"
        elif quality_score >= 80:
            return "高"
        elif quality_score >= 70:
            return "中等"
        elif quality_score >= 60:
            return "较低"
        else:
            return "低"
    
    def get_model_status(self):
        """获取模型状态"""
        return {
            'available': self.check_availability(),
            'loaded': self.loaded,
            'device': self.device,
            'model_info': self.model_info if self.loaded else None
        }


# 全局模型服务实例
model_service = KronosModelService()


def get_model_service():
    """获取模型服务实例"""
    return model_service
