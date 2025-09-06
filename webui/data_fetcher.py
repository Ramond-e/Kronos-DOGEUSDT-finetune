#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DOGE/USDT WebUI 数据获取和预处理脚本
功能：获取最新400小时的DOGE/USDT永续合约1小时K线数据，预处理后保存为Kronos可用格式
"""

import os
import sys
import time
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from binance import Client
import csv


def ms2str(ms):
    """将毫秒时间戳转换为可读字符串"""
    from datetime import timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_klines_with_retry(client, params, max_retries=3):
    """带重试机制的USDⓈ-M合约K线数据获取"""
    for attempt in range(max_retries):
        try:
            return client.futures_klines(**params)
        except Exception as e:
            print(f"  第 {attempt + 1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print("  达到最大重试次数，数据获取失败")
                raise e


def fetch_latest_klines(hours=400):
    """
    获取最新指定小时数的DOGE/USDT永续合约K线数据
    
    Args:
        hours (int): 获取的小时数，默认400
        
    Returns:
        list: K线数据列表
    """
    print(f"🔄 开始获取DOGE/USDT最新{hours}小时K线数据...")
    
    client = Client()
    symbol = "DOGEUSDT"
    interval = "1h"
    
    all_klines = []
    processed_times = set()  # 用于避免重复数据
    target_count = hours
    
    # 从当前时间开始，向过去获取
    end_time = int(time.time() * 1000)
    
    while len(all_klines) < target_count:
        # 计算还需要多少条数据
        remaining = target_count - len(all_klines)
        limit = min(1500, remaining)  # 每次最多1500条
        
        try:
            params = {
                "symbol": symbol, 
                "interval": interval, 
                "limit": limit, 
                "endTime": end_time
            }
            
            klines = get_klines_with_retry(client, params)
            
            if not klines:
                print("  没有更多数据可获取")
                break
            
            # 过滤重复数据
            new_klines = []
            for kline in klines:
                if kline[0] not in processed_times:
                    new_klines.append(kline)
                    processed_times.add(kline[0])
            
            if not new_klines:
                print("  所有数据都已处理过")
                break
                
            # 添加新数据（注意：API返回的是倒序，最新的在前）
            all_klines.extend(new_klines)
            print(f"  已获取 {len(all_klines)}/{target_count} 条数据")
            
            # 设置下一轮的结束时间为当前批次最早的时间-1
            first_open_time = new_klines[-1][0]  # 最早的一条
            end_time = first_open_time - 1
            
            # 如果这次返回的数据不够limit，说明没有更多历史数据
            if len(klines) < limit:
                print("  已到达最早历史数据")
                break
                
            # 添加延迟避免API限制
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  获取数据时发生错误: {e}")
            raise e
    
    if all_klines:
        # 按时间升序排序（从过去到现在）
        all_klines.sort(key=lambda x: x[0])
        print(f"✅ 成功获取 {len(all_klines)} 条K线数据")
        print(f"  时间范围: {ms2str(all_klines[0][0])} 到 {ms2str(all_klines[-1][0])}")
        return all_klines
    else:
        raise Exception("没有获取到任何K线数据")


def preprocess_data(raw_data):
    """
    预处理K线数据，转换为Kronos需要的格式
    
    Args:
        raw_data (list): 原始K线数据
        
    Returns:
        pd.DataFrame: 处理后的数据框
    """
    print("🔄 开始数据预处理...")
    
    # 转换为DataFrame
    df = pd.DataFrame(raw_data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # 转换数据类型
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    df['quote_asset_volume'] = df['quote_asset_volume'].astype(float)
    
    # 重命名为Kronos期望的格式，使用Binance真实成交额
    df = df.rename(columns={
        'open_time': 'timestamps',
        'quote_asset_volume': 'amount'
    })
    
    # 选择Kronos需要的列
    df_processed = df[['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
    
    # 验证数据完整性
    if df_processed.isnull().any().any():
        print("  ⚠️ 检测到缺失值，进行处理...")
        df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ 数据预处理完成，最终数据形状: {df_processed.shape}")
    print("  列名:", df_processed.columns.tolist())
    
    return df_processed


def generate_timestamped_filename():
    """生成带时间戳的文件名"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"dogeusdt_{timestamp}.csv"


def save_processed_data(df, data_dir="data"):
    """
    保存处理后的数据到指定目录
    
    Args:
        df (pd.DataFrame): 处理后的数据框
        data_dir (str): 数据目录路径
        
    Returns:
        str: 保存的文件路径
    """
    print("🔄 保存处理后的数据...")
    
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)
    
    # 生成时间戳文件名
    filename = generate_timestamped_filename()
    filepath = os.path.join(data_dir, filename)
    
    # 保存为CSV
    df.to_csv(filepath, index=False, encoding='utf-8')
    
    print(f"✅ 数据已保存到: {filepath}")
    print(f"  文件大小: {os.path.getsize(filepath) / 1024:.2f} KB")
    
    return filepath


def get_latest_data_file(data_dir="data"):
    """
    获取数据目录中最新的数据文件
    
    Args:
        data_dir (str): 数据目录路径
        
    Returns:
        str: 最新数据文件路径，如果没有文件则返回None
    """
    pattern = os.path.join(data_dir, "dogeusdt_*.csv")
    data_files = glob.glob(pattern)
    
    if not data_files:
        return None
    
    # 按文件创建时间排序，返回最新的
    latest_file = max(data_files, key=os.path.getctime)
    return latest_file


def validate_data_file(filepath):
    """
    验证数据文件的有效性
    
    Args:
        filepath (str): 数据文件路径
        
    Returns:
        bool: 文件是否有效
    """
    try:
        df = pd.read_csv(filepath)
        
        # 检查必需的列
        required_cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume', 'amount']
        if not all(col in df.columns for col in required_cols):
            return False
        
        # 检查数据量
        if len(df) < 100:  # 至少要有100条数据
            return False
            
        # 检查数据类型
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        return True
    except Exception:
        return False


def main():
    """主函数"""
    # 设置编码以避免Windows命令行显示问题
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("DOGE/USDT 数据获取和预处理工具")
    print("=" * 50)
    
    try:
        # 1. 获取最新400小时数据
        raw_data = fetch_latest_klines(hours=400)
        
        # 2. 数据预处理
        processed_df = preprocess_data(raw_data)
        
        # 3. 保存数据
        filepath = save_processed_data(processed_df)
        
        # 4. 验证保存的文件
        if validate_data_file(filepath):
            print("✅ 数据文件验证通过")
        else:
            print("❌ 数据文件验证失败")
            return
        
        # 5. 显示摘要信息
        print("\n📊 数据摘要:")
        print(f"  数据条数: {len(processed_df)}")
        print(f"  时间范围: {processed_df['timestamps'].iloc[0]} 到 {processed_df['timestamps'].iloc[-1]}")
        print(f"  价格范围: {processed_df['close'].min():.6f} - {processed_df['close'].max():.6f}")
        print(f"  平均交易量: {processed_df['volume'].mean():.2f}")
        
        # 6. 显示最新数据文件信息
        latest_file = get_latest_data_file()
        if latest_file:
            print(f"\n📁 最新数据文件: {latest_file}")
        
        print("\n🎉 数据获取和预处理完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
