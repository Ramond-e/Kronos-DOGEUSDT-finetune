#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Kronos DOGE/USDT WebUI 启动脚本
一键启动完整的预测系统
"""

import os
import sys
import time
import subprocess
from datetime import datetime

def print_banner():
    """打印启动横幅"""
    print("=" * 70)
    print("🚀 Kronos DOGE/USDT 预测系统启动器")
    print("=" * 70)
    print("📊 基于微调Kronos模型的高精度DOGE/USDT价格趋势预测")
    print("🔮 400小时历史数据 + 120小时未来预测")
    print("💡 专业级预测质量控制和趋势分析")
    print("=" * 70)
    print()

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python版本过低: {python_version.major}.{python_version.minor}")
        print("   请使用Python 3.8及以上版本")
        return False
    
    print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 检查必需模块
    required_modules = [
        'flask', 'pandas', 'numpy', 'torch', 'binance'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}: 已安装")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module}: 未安装")
    
    if missing_modules:
        print(f"\n⚠️ 缺少必需模块: {', '.join(missing_modules)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    # 检查模型模块
    try:
        sys.path.append('../')
        from model.kronos import KronosTokenizer, Kronos, KronosPredictor
        print("✅ Kronos模型模块: 可用")
    except ImportError as e:
        print(f"❌ Kronos模型模块: 不可用 ({e})")
        print("   请检查model目录和相关依赖")
        return False
    
    print("✅ 环境检查通过!\n")
    return True

def check_data_files():
    """检查数据文件"""
    print("📁 检查数据文件...")
    
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📂 创建数据目录: {data_dir}")
    
    # 查找数据文件
    data_files = []
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.startswith('dogeusdt_') and file.endswith('.csv'):
                data_files.append(file)
    
    if data_files:
        data_files.sort(reverse=True)  # 最新的在前
        latest_file = data_files[0]
        print(f"✅ 找到 {len(data_files)} 个数据文件")
        print(f"📊 最新文件: {latest_file}")
        
        # 检查文件大小和修改时间
        file_path = os.path.join(data_dir, latest_file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"📏 文件大小: {file_size:.1f}KB")
        print(f"⏰ 修改时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查文件是否过旧（超过24小时）
        now = datetime.now()
        if (now - mod_time).total_seconds() > 24 * 3600:
            print("⚠️ 数据文件超过24小时，建议更新")
            return False, True  # 有文件但过旧
        else:
            print("✅ 数据文件较新")
            return True, False
    else:
        print("❌ 未找到数据文件")
        return False, False

def fetch_data():
    """获取最新数据"""
    print("\n📡 获取最新DOGE/USDT数据...")
    print("⏳ 正在从Binance获取过去400小时的K线数据...")
    
    try:
        # 运行数据获取脚本
        result = subprocess.run([sys.executable, 'data_fetcher.py'], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ 数据获取成功!")
            # 显示部分输出
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[-5:]:  # 显示最后5行
                if line.strip():
                    print(f"   {line}")
            return True
        else:
            print(f"❌ 数据获取失败:")
            print(f"   {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 数据获取超时（5分钟）")
        return False
    except Exception as e:
        print(f"❌ 数据获取错误: {e}")
        return False

def start_web_server():
    """启动Web服务器"""
    print("\n🌐 启动Kronos预测Web服务器...")
    print("🔗 访问地址: http://127.0.0.1:5000")
    print("🎯 功能包括:")
    print("   • 📡 实时数据获取：每次预测前自动获取最新400小时数据")
    print("   • 🔮 高精度价格趋势预测：预测未来120小时价格走势")
    print("   • 🎛️ 专业级预测质量控制：Temperature、top_p等参数调节")
    print("   • 📈 交互式K线图表展示：Plotly.js专业金融图表")
    print("   • 🧠 智能趋势和波动性分析：5级趋势分类系统")
    print("\n💡 使用提示:")
    print("   1. 等待页面加载完成")
    print("   2. 📡 推荐开启'自动获取最新数据'获得最佳预测效果")
    print("   3. 🎛️ 调整预测参数（Temperature=1.0, top_p=0.9推荐）")
    print("   4. 🔮 点击'开始预测'自动获取数据并预测")
    print("   5. 📈 查看预测结果和趋势分析")
    print("\n🛑 按 Ctrl+C 停止服务")
    print("=" * 70)
    
    try:
        # 导入并启动Flask应用
        from app import app
        app.run(host='127.0.0.1', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 Kronos预测系统已停止")
        print("感谢使用！")
    except Exception as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)

def main():
    """主函数"""
    print_banner()
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请解决上述问题后重试")
        sys.exit(1)
    
    # 检查数据文件
    has_data, is_old = check_data_files()
    
    if not has_data:
        print("💡 点击'开始预测'时会自动获取最新400小时数据")
    elif is_old:
        print("💡 建议开启'自动获取最新数据'以获得最佳预测效果")
    
    # 启动Web服务器
    start_web_server()

if __name__ == '__main__':
    main()
