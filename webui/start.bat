
@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
cls

echo ======================================================================
echo 🚀 Kronos DOGE/USDT 预测系统启动器 (Windows)
echo ======================================================================
echo 📊 基于微调Kronos模型的高精度DOGE/USDT价格趋势预测
echo 🔮 400小时历史数据 + 120小时未来预测
echo 💡 专业级预测质量控制和趋势分析
echo ======================================================================
echo.

echo 🔍 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 未安装，请先安装Python 3.8+
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%i
echo ✅ Python版本: %PYTHON_VERSION%

echo.
echo 📁 检查项目文件...
if not exist "app.py" (
    echo ❌ 请在webui目录下运行此脚本
    echo    cd path\to\kronos\webui
    echo    start.bat
    pause
    exit /b 1
)

if not exist "data" (
    echo 📂 创建数据目录...
    mkdir data
)

echo.
echo 📦 检查依赖包...
if exist "requirements.txt" (
    echo ✅ 找到requirements.txt
    echo 💡 如果遇到导入错误，请运行: pip install -r requirements.txt
) else (
    echo ⚠️ 未找到requirements.txt
)

echo.
echo 📁 检查数据文件...
set DATA_COUNT=0
for %%f in (data\dogeusdt_*.csv) do set /a DATA_COUNT+=1

if %DATA_COUNT%==0 (
    echo ⚠️ 未找到历史数据文件
    echo.
    set /p choice="🤔 是否现在获取数据? (y/n，可跳过): "
    if /i "!choice!"=="y" (
        echo 📡 获取DOGE/USDT数据...
        python data_fetcher.py
        if errorlevel 1 (
            echo ❌ 数据获取失败，Web界面仍可使用实时获取功能
        ) else (
            echo ✅ 数据获取成功!
        )
    ) else (
        echo 💡 已跳过，Web界面可使用实时数据获取功能
    )
) else (
    echo ✅ 找到 %DATA_COUNT% 个历史数据文件
)

echo.
echo 🌐 启动Kronos预测Web服务器...
echo 🔗 访问地址: http://127.0.0.1:5000
echo 🎯 功能包括:
echo    • 实时数据获取：每次预测前自动获取最新400小时数据
echo    • 高精度价格趋势预测：预测未来120小时价格走势
echo    • 专业级预测质量控制：Temperature、top_p等参数设置
echo    • 交互式K线：Plotly.js专业金融可视化
echo    • 智能趋势和波动性分析：5级趋势分类系统
echo.
echo 使用提示:
echo    1. 等待页面加载完成
echo    2. 推荐开启自动获取最新数据获得最佳预测效果
echo    3. 设置预测参数 Temperature=1.0 top_p=0.9
echo    4. 点击开始预测按钮执行数据获取和预测
echo    5. 检视预测结果和趋势分析
echo.
echo 按 Ctrl+C 停止服务
echo ======================================================================

python run.py

echo.
echo Kronos预测系统已停止
echo 使用结束！
pause
