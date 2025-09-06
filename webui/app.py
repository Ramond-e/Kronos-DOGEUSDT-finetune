#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kronos DOGE/USDT WebUI Flask应用
简化版本：专注于核心预测功能
"""

import os
import sys
import json
import traceback
from datetime import datetime
import pytz
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
sys.path.append('../')

# 导入本地模块
from data_fetcher import get_latest_data_file, validate_data_file
from model_service import get_model_service

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 支持中文JSON

# 获取模型服务实例
model_service = get_model_service()


def format_time_beijing(timestamp_str):
    """将时间戳转换为北京时间显示"""
    try:
        dt = pd.to_datetime(timestamp_str)
        if dt.tz is None:
            dt = dt.tz_localize('UTC')
        beijing_tz = pytz.timezone('Asia/Shanghai')
        beijing_time = dt.tz_convert(beijing_tz)
        return beijing_time.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return timestamp_str


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/status')
def get_status():
    """获取系统状态"""
    try:
        # 检查数据文件状态
        latest_file = get_latest_data_file()
        data_status = {
            'available': latest_file is not None,
            'file_path': latest_file,
            'valid': validate_data_file(latest_file) if latest_file else False
        }
        
        if latest_file:
            # 获取数据文件信息
            df = pd.read_csv(latest_file)
            data_status.update({
                'rows': len(df),
                'time_range': {
                    'start': format_time_beijing(df['timestamps'].iloc[0]),
                    'end': format_time_beijing(df['timestamps'].iloc[-1])
                },
                'file_size_kb': round(os.path.getsize(latest_file) / 1024, 2)
            })
        
        # 检查模型状态
        model_status = model_service.get_model_status()
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': data_status,
            'model': model_status
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/latest-data')
def get_latest_data_info():
    """获取最新数据文件信息"""
    try:
        latest_file = get_latest_data_file()
        
        if not latest_file:
            return jsonify({
                'success': False,
                'error': '没有找到数据文件，请先运行data_fetcher.py获取数据'
            }), 404
        
        if not validate_data_file(latest_file):
            return jsonify({
                'success': False,
                'error': '数据文件无效，请重新获取数据'
            }), 400
        
        # 读取数据文件信息
        df = pd.read_csv(latest_file)
        
        # 基本信息
        data_info = {
            'file_path': latest_file,
            'file_name': os.path.basename(latest_file),
            'rows': len(df),
            'columns': df.columns.tolist(),
            'file_size_kb': round(os.path.getsize(latest_file) / 1024, 2),
            'time_range': {
                'start': format_time_beijing(df['timestamps'].iloc[0]),
                'end': format_time_beijing(df['timestamps'].iloc[-1])
            },
            'price_range': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max())
            },
            'avg_volume': float(df['volume'].mean())
        }
        
        return jsonify({
            'success': True,
            'data': data_info,
            'message': f'数据文件加载成功：{len(df)}条记录'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'获取数据信息失败: {str(e)}'
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """执行预测"""
    try:
        # 获取预测参数
        data = request.get_json() or {}
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        auto_fetch = data.get('auto_fetch', True)  # 默认自动获取最新数据
        
        print(f"🔮 接收预测请求: T={temperature}, top_p={top_p}, samples={sample_count}")
        print(f"📡 自动获取最新数据: {auto_fetch}")
        
        # 检查模型可用性
        if not model_service.check_availability():
            return jsonify({
                'success': False,
                'error': 'Kronos模型不可用，请检查model模块安装'
            }), 500
        
        # 加载模型（如果未加载）
        if not model_service.loaded:
            print("🔄 首次使用，正在加载模型...")
            if not model_service.load_model():
                return jsonify({
                    'success': False,
                    'error': '模型加载失败，请检查网络连接和Hugging Face访问'
                }), 500
        
        # 获取最新数据
        if auto_fetch:
            print("📡 正在获取最新DOGE/USDT数据...")
            from data_fetcher import fetch_latest_klines, preprocess_data, save_processed_data
            
            try:
                # 获取最新400小时数据
                raw_data = fetch_latest_klines(hours=400)
                if not raw_data:
                    return jsonify({
                        'success': False,
                        'error': '获取最新数据失败，请检查网络连接'
                    }), 500
                
                # 预处理数据
                df = preprocess_data(raw_data)
                print(f"✅ 成功获取最新数据: {len(df)}条记录")
                
                # 保存最新数据（可选，用于调试）
                latest_file = save_processed_data(df)
                print(f"💾 数据已保存到: {latest_file}")
                
            except Exception as e:
                print(f"❌ 获取最新数据失败: {e}")
                # 回退到使用历史数据文件
                print("🔄 回退使用历史数据文件...")
                latest_file = get_latest_data_file()
                if not latest_file:
                    return jsonify({
                        'success': False,
                        'error': '获取最新数据失败，且没有历史数据可用'
                    }), 500
                
                df = pd.read_csv(latest_file)
                print(f"📊 使用历史数据文件: {latest_file}")
        else:
            # 使用历史数据文件
            latest_file = get_latest_data_file()
            if not latest_file:
                return jsonify({
                    'success': False,
                    'error': '没有找到数据文件，请先运行data_fetcher.py获取数据'
                }), 404
            
            if not validate_data_file(latest_file):
                return jsonify({
                    'success': False,
                    'error': '数据文件无效，请重新获取数据'
                }), 400
            
            print("📊 使用历史数据文件...")
            df = pd.read_csv(latest_file)
        
        # 数据验证
        if len(df) < 400:
            return jsonify({
                'success': False,
                'error': f'数据不足：需要至少400条数据，当前只有{len(df)}条'
            }), 400
        
        # 使用专业模型服务进行预测
        print("🚀 开始高精度预测...")
        prediction_result = model_service.predict_future(
            data_df=df,
            pred_hours=120,
            temperature=temperature,
            top_p=top_p,
            sample_count=sample_count
        )
        
        print("✅ 预测完成!")
        
        # 返回结果
        return jsonify({
            'success': True,
            'message': f'高精度预测完成：基于{prediction_result["parameters"]["historical_hours"]}小时历史数据预测未来120小时',
            'data': {
                'chart_data': prediction_result['chart_data'],
                'trend_analysis': prediction_result['trend_analysis'],
                'volatility_analysis': prediction_result['volatility_analysis'],
                'quality_metrics': prediction_result['quality_metrics'],
                'parameters': prediction_result['parameters'],
                'model_info': prediction_result['model_info'],
                'generated_at': prediction_result['generated_at']
            }
        })
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'参数错误: {str(e)}'
        }), 400
        
    except Exception as e:
        error_msg = f"预测失败: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500




@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({'success': False, 'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({'success': False, 'error': '服务器内部错误'}), 500


if __name__ == '__main__':
    print("🚀 启动Kronos DOGE/USDT WebUI")
    print("=" * 50)
    
    # 检查数据文件
    latest_file = get_latest_data_file()
    if latest_file:
        print(f"✅ 找到数据文件: {latest_file}")
    else:
        print("⚠️ 未找到数据文件，请先运行 python data_fetcher.py")
    
    # 检查模型可用性
    if model_service.check_availability():
        print("✅ Kronos模型可用")
    else:
        print("⚠️ Kronos模型不可用，请检查model模块")
    
    print("\n🌐 启动Flask应用...")
    print("访问地址: http://127.0.0.1:5000")
    print("按 Ctrl+C 停止服务")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
