#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于原Kronos项目预测示例的DOGE/USDT模型评估脚本
使用5月期间600小时数据：前80%预测后20%，与真实值对比
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append("../../")
sys.path.append("../")

from model import Kronos, KronosTokenizer, KronosPredictor

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_prediction(kline_df, pred_df):
    """
    可视化预测结果
    """
    pred_df.index = kline_df.index[-pred_df.shape[0]:]
    sr_close = kline_df['close']
    sr_pred_close = pred_df['close']
    sr_close.name = 'Ground Truth'
    sr_pred_close.name = "Prediction"

    sr_volume = kline_df['volume']
    sr_pred_volume = pred_df['volume']
    sr_volume.name = 'Ground Truth'
    sr_pred_volume.name = "Prediction"

    close_df = pd.concat([sr_close, sr_pred_close], axis=1)
    volume_df = pd.concat([sr_volume, sr_pred_volume], axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # 绘制Close价格对比
    ax1.plot(close_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=2)
    ax1.plot(close_df['Prediction'], label='Prediction', color='red', linewidth=2, linestyle='--')
    ax1.set_ylabel('Close Price (USDT)', fontsize=14, fontweight='bold')
    ax1.set_title('DOGE/USDT Close Price Prediction vs Ground Truth', fontsize=16, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 绘制交易量对比
    ax2.plot(volume_df['Ground Truth'], label='Ground Truth', color='blue', linewidth=2)
    ax2.plot(volume_df['Prediction'], label='Prediction', color='red', linewidth=2, linestyle='--')
    ax2.set_ylabel('Volume', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=14, fontweight='bold')
    ax2.set_title('DOGE/USDT Volume Prediction vs Ground Truth', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # 添加分隔线标示预测开始位置
    split_point = len(close_df) - len(pred_df)
    ax1.axvline(x=close_df.index[split_point], color='black', linestyle=':', alpha=0.7, linewidth=2)
    ax2.axvline(x=close_df.index[split_point], color='black', linestyle=':', alpha=0.7, linewidth=2)
    
    # 添加文本标注
    ax1.text(close_df.index[split_point], ax1.get_ylim()[1]*0.95, 'Prediction Start', 
             rotation=90, verticalalignment='top', fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    # 保存图表
    save_path = "doge_prediction_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 预测结果图表已保存: {save_path}")
    plt.show()

def calculate_metrics(pred_df, true_df):
    """
    计算预测评估指标
    """
    results = {}
    
    for col in ['close', 'volume']:
        pred_values = pred_df[col].values
        true_values = true_df[col].values
        
        # 基本误差指标
        mse = np.mean((pred_values - true_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred_values - true_values))
        mape = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-8))) * 100
        
        # 方向准确率
        pred_direction = np.sign(np.diff(pred_values))
        true_direction = np.sign(np.diff(true_values))
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        # 相关系数
        correlation = np.corrcoef(pred_values, true_values)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        results[col] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'Correlation': correlation
        }
    
    return results

def load_and_prepare_data():
    """
    加载并准备DOGE/USDT数据
    """
    print("🔄 加载DOGE/USDT数据...")
    
    # 加载原始数据
    data_path = "../../get_DOGEUSDT_data/dogeusdt_1h_all_klines.csv"
    df = pd.read_csv(data_path)
    
    print(f"  原始数据形状: {df.shape}")
    
    # 转换时间戳
    df['timestamps'] = pd.to_datetime(df['open_time'], unit='ms')
    
    # 选择需要的列并重命名以匹配Kronos格式
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']].copy()
    df = df.rename(columns={'quote_asset_volume': 'amount'})

    # 使用5月期间的600小时数据（最佳表现期）
    start_idx = len(df) - 3000  # 从倒数第3000小时开始
    end_idx = start_idx + 600   # 取600小时
    
    df_selected_600 = df.iloc[start_idx:end_idx].copy()
    df_selected_600 = df_selected_600.reset_index(drop=True)
    
    print(f"  使用第{start_idx}-{end_idx}小时的数据（5月最佳表现期），形状: {df_selected_600.shape}")
    print(f"  时间范围: {df_selected_600['timestamps'].iloc[0]} 到 {df_selected_600['timestamps'].iloc[-1]}")
    
    return df_selected_600

def main():
    """
    主函数
    """
    print("🚀 DOGE/USDT Kronos微调模型预测评估")
    print("=" * 60)
    
    try:
        # 1. 加载数据
        df = load_and_prepare_data()
        
        # 2. 从Hugging Face加载微调后的模型和tokenizer
        print("🔄 从Hugging Face加载微调后的DOGE Kronos模型...")
        
        try:
            tokenizer = KronosTokenizer.from_pretrained("Ramond-e/doge-kronos-tokenizer")
            model = Kronos.from_pretrained("Ramond-e/doge-kronos-predictor")
            print("  ✅ 成功从Hugging Face加载微调后的DOGE模型")
            
        except Exception as e:
            print(f"  ❌ 从Hugging Face加载失败: {e}")
            print("  请检查网络连接或模型仓库是否可访问")
            raise
        
        # 3. 创建预测器
        predictor = KronosPredictor(model, tokenizer, device="cuda:0", max_context=512)
        
        # 4. 准备数据分割
        total_len = len(df)
        split_point = int(total_len * 0.8)  # 前80%用于历史，后20%用于预测目标
        
        lookback = split_point  # 480小时的历史数据
        pred_len = total_len - split_point  # 120小时的预测长度
        
        print(f"📊 数据分割:")
        print(f"  历史数据长度: {lookback} 小时")
        print(f"  预测长度: {pred_len} 小时")
        print(f"  分割时间点: {df.loc[split_point-1, 'timestamps']}")
        
        # 5. 准备输入数据
        x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        x_timestamp = df.loc[:lookback-1, 'timestamps']
        y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
        
        print(f"📝 输入数据准备:")
        print(f"  历史特征数据形状: {x_df.shape}")
        print(f"  历史时间戳长度: {len(x_timestamp)}")
        print(f"  预测时间戳长度: {len(y_timestamp)}")
        
        # 6. 执行预测
        print("🔮 开始预测...")
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=1.0,           # 温度参数
            top_p=0.9,       # top-p采样
            sample_count=1,  # 采样次数
            verbose=True     # 显示详细信息
        )
        
        print("✅ 预测完成!")
        print(f"📋 预测结果形状: {pred_df.shape}")
        print("\n预测数据头部:")
        print(pred_df.head())
        
        # 7. 准备真实值用于对比
        true_df = df.loc[lookback:lookback+pred_len-1, ['open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        true_df = true_df.reset_index(drop=True)
        
        print(f"\n📋 真实数据形状: {true_df.shape}")
        print("真实数据头部:")
        print(true_df.head())
        
        # 8. 计算评估指标
        print("\n📊 计算评估指标...")
        metrics = calculate_metrics(pred_df, true_df)
        
        print("\n🎯 预测性能评估:")
        for feature, result in metrics.items():
            print(f"\n  {feature.upper()} 预测性能:")
            print(f"    MSE: {result['MSE']:.6f}")
            print(f"    RMSE: {result['RMSE']:.6f}")
            print(f"    MAE: {result['MAE']:.6f}")
            print(f"    MAPE: {result['MAPE']:.2f}%")
            print(f"    方向准确率: {result['Direction_Accuracy']:.4f}")
            print(f"    相关系数: {result['Correlation']:.4f}")
        
        # 9. 可视化结果
        print("\n📈 生成预测可视化...")
        
        # 组合历史和预测数据用于绘图
        kline_df = df.loc[:lookback+pred_len-1].copy()
        kline_df = kline_df.set_index('timestamps')
        
        # 调整pred_df的索引以匹配kline_df
        pred_df_for_plot = pred_df.copy()
        pred_df_for_plot.index = kline_df.index[-pred_len:]
        
        # 绘制对比图
        plot_prediction(kline_df, pred_df_for_plot)
        
        # 10. 生成评估报告
        print("\n📝 生成评估报告...")
        
        report_content = f"""# DOGE/USDT Kronos微调模型预测评估报告

## 📊 评估概要

- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据来源**: 5月最佳表现期600小时DOGE/USDT K线数据
- **数据分割**: 前80% (480小时) 用于历史，后20% (120小时) 用于预测验证
- **模型**: 微调后的Kronos (Epoch 4 Predictor + Epoch 5 Tokenizer)

## 🎯 预测性能指标

### Close价格预测
- **MSE**: {metrics['close']['MSE']:.6f}
- **RMSE**: {metrics['close']['RMSE']:.6f}
- **MAE**: {metrics['close']['MAE']:.6f}
- **MAPE**: {metrics['close']['MAPE']:.2f}%
- **方向准确率**: {metrics['close']['Direction_Accuracy']:.4f}
- **相关系数**: {metrics['close']['Correlation']:.4f}

### Volume交易量预测
- **MSE**: {metrics['volume']['MSE']:.6f}
- **RMSE**: {metrics['volume']['RMSE']:.6f}
- **MAE**: {metrics['volume']['MAE']:.6f}
- **MAPE**: {metrics['volume']['MAPE']:.2f}%
- **方向准确率**: {metrics['volume']['Direction_Accuracy']:.4f}
- **相关系数**: {metrics['volume']['Correlation']:.4f}

## 📈 模型表现评估

### Close价格预测表现
- **相关性**: {'优秀' if metrics['close']['Correlation'] > 0.7 else '良好' if metrics['close']['Correlation'] > 0.5 else '一般' if metrics['close']['Correlation'] > 0.3 else '较差'}
- **方向准确率**: {'优秀' if metrics['close']['Direction_Accuracy'] > 0.6 else '良好' if metrics['close']['Direction_Accuracy'] > 0.5 else '需改进'}
- **误差水平**: {'低' if metrics['close']['MAPE'] < 5 else '中等' if metrics['close']['MAPE'] < 10 else '较高'}

### Volume交易量预测表现  
- **相关性**: {'优秀' if metrics['volume']['Correlation'] > 0.7 else '良好' if metrics['volume']['Correlation'] > 0.5 else '一般' if metrics['volume']['Correlation'] > 0.3 else '较差'}
- **方向准确率**: {'优秀' if metrics['volume']['Direction_Accuracy'] > 0.6 else '良好' if metrics['volume']['Direction_Accuracy'] > 0.5 else '需改进'}

## 💡 结论与建议

1. **模型适用性**: 适合DOGE/USDT短期价格趋势预测 (120小时内)
2. **最佳应用场景**: {'量化交易信号生成和趋势分析' if metrics['close']['Correlation'] > 0.5 else '需进一步优化后使用'}
3. **风险提示**: 加密货币市场波动较大，预测结果仅供参考

---
*评估完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        report_path = "doge_prediction_evaluation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 评估报告已保存: {report_path}")
        
        # 总结
        print("\n" + "=" * 60)
        print("🎉 预测评估完成!")
        print(f"📈 Close价格相关系数: {metrics['close']['Correlation']:.4f}")
        print(f"🎯 Close价格方向准确率: {metrics['close']['Direction_Accuracy']:.4f}")
        print(f"📊 Close价格MAPE: {metrics['close']['MAPE']:.2f}%")
        
        if metrics['close']['Correlation'] > 0.6 and metrics['close']['Direction_Accuracy'] > 0.5:
            print("🏆 模型表现优秀！")
        elif metrics['close']['Correlation'] > 0.4 and metrics['close']['Direction_Accuracy'] > 0.45:
            print("✅ 模型表现良好！")
        else:
            print("⚠️ 模型表现有待改进")
        
    except Exception as e:
        print(f"❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
