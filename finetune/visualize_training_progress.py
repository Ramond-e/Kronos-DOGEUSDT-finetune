#!/usr/bin/env python3
"""
DOGE/USDT交易对训练进度可视化脚本
展示predictor的损失变化
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_progress():
    """绘制Predictor训练进度图表"""
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Predictor训练数据 (7轮)
    predictor_epochs = [1, 2, 3, 4, 5, 6, 7]
    predictor_train_loss = [1.946293, 1.751143, 1.706408, 1.680686, 1.663046, 1.649754, 1.639570]
    predictor_val_loss = [2.021497, 1.986604, 1.974955, 1.971007, 1.971090, 1.972143, 1.974842]
    
    # 创建单个图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 绘制训练损失和验证损失曲线
    train_line = ax.plot(predictor_epochs, predictor_train_loss, 'o-', linewidth=3, markersize=10, 
                        color='#2E86AB', label='训练损失', markerfacecolor='white', markeredgewidth=2)
    val_line = ax.plot(predictor_epochs, predictor_val_loss, 's-', linewidth=3, markersize=10, 
                      color='#C73E1D', label='验证损失', markerfacecolor='white', markeredgewidth=2)
    
    # 设置标题和标签
    ax.set_title('DOGE/USDT交易对 Predictor训练损失变化', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('训练轮次 (Epoch)', fontsize=14)
    ax.set_ylabel('损失值 (Loss)', fontsize=14)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 找出最佳轮次
    best_epoch = predictor_val_loss.index(min(predictor_val_loss)) + 1
    best_val_loss = min(predictor_val_loss)
    
    # 标注每个数据点的数值
    for i, (epoch, train_loss, val_loss) in enumerate(zip(predictor_epochs, predictor_train_loss, predictor_val_loss)):
        # 训练损失标注
        ax.annotate(f'{train_loss:.3f}', 
                   xy=(epoch, train_loss), 
                   xytext=(0, -20), textcoords='offset points',
                   ha='center', va='top', fontsize=10, color='#2E86AB', fontweight='bold')
        
        # 验证损失标注
        ax.annotate(f'{val_loss:.3f}', 
                   xy=(epoch, val_loss), 
                   xytext=(0, 15), textcoords='offset points',
                   ha='center', va='bottom', fontsize=10, color='#C73E1D', fontweight='bold')
    
    # 标注最佳点
    ax.annotate(f'★ 最佳模型\n轮次 {best_epoch}', 
                xy=(best_epoch, best_val_loss), 
                xytext=(best_epoch+1.2, best_val_loss-0.03),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, color='red', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    # 只标注过拟合区域 (5-7轮) - 训练损失继续下降，验证损失开始上升
    ax.axvspan(4.5, 7.5, alpha=0.2, color='red', label='过拟合区域')
    ax.text(6, 1.68, '过拟合区域', fontsize=14, fontweight='bold', color='red', 
            ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    # 添加泛化差距线 (验证损失与训练损失的差距)
    # 在第4轮后添加警戒线，显示差距开始增大
    for i, epoch in enumerate(predictor_epochs[4:], 5):
        diff = predictor_val_loss[i-1] - predictor_train_loss[i-1]
        if diff > 0.32:  # 当差距超过0.32时，用虚线连接
            ax.plot([epoch, epoch], [predictor_train_loss[i-1], predictor_val_loss[i-1]], 
                   'r--', alpha=0.6, linewidth=2)
    
    # 添加趋势箭头
    # 训练损失趋势箭头
    ax.annotate('', xy=(7, predictor_train_loss[-1]), xytext=(6, predictor_train_loss[-2]),
                arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=3))
    ax.text(6.5, predictor_train_loss[-1]-0.01, '继续下降', fontsize=10, color='#2E86AB', 
            ha='center', fontweight='bold')
    
    # 验证损失趋势箭头
    ax.annotate('', xy=(7, predictor_val_loss[-1]), xytext=(6, predictor_val_loss[-2]),
                arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=3))
    ax.text(6.5, predictor_val_loss[-1]+0.01, '开始上升', fontsize=10, color='#C73E1D', 
            ha='center', fontweight='bold')
    
    # 设置图例位置到左下角，避免与图片重合
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9, 
              bbox_to_anchor=(0.02, 0.02))
    
    # 设置坐标轴范围，给数值标注留出空间
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(1.58, 2.07)
    
    # 美化坐标轴
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = 'doge_outputs/predictor_training_progress.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Predictor训练进度图表已保存: {save_path}")
    
    plt.show()

def print_training_summary():
    """打印训练总结"""
    print("=" * 80)
    print("🎯 DOGE/USDT交易对模型训练总结")
    print("=" * 80)
    
    print("\n📈 DOGE/USDT Predictor训练 (7轮):")
    print("   轮次 1: 训练损失=1.946293, 验证损失=2.021497")
    print("   轮次 2: 训练损失=1.751143, 验证损失=1.986604")
    print("   轮次 3: 训练损失=1.706408, 验证损失=1.974955")
    print("   轮次 4: 训练损失=1.680686, 验证损失=1.971007 ⭐ 最佳")
    print("   轮次 5: 训练损失=1.663046, 验证损失=1.971090")
    print("   轮次 6: 训练损失=1.649754, 验证损失=1.972143")
    print("   轮次 7: 训练损失=1.639570, 验证损失=1.974842")
    
    print("\n🎯 关键观察:")
    print("   • 第4轮达到最佳验证性能 (验证损失: 1.971007)")
    print("   • 从第5轮开始出现过拟合现象")
    print("   • 训练损失持续下降，验证损失开始上升")
    print("   • 最佳模型性能稳定，适合部署使用")
    
    print("\n💾 已保存模型:")
    print("   • best_predictor_epoch_4.pt ✅ (最佳DOGE/USDT预测模型)")
    
    print("=" * 80)

def analyze_overfitting():
    """分析过拟合情况"""
    print("\n🔍 过拟合分析:")
    
    predictor_train_loss = [1.946293, 1.751143, 1.706408, 1.680686, 1.663046, 1.649754, 1.639570]
    predictor_val_loss = [2.021497, 1.986604, 1.974955, 1.971007, 1.971090, 1.972143, 1.974842]
    
    print("   轮次  |  训练损失  |  验证损失  |  差异   |  状态")
    print("   -----|-----------|-----------|---------|--------")
    
    for i, (train, val) in enumerate(zip(predictor_train_loss, predictor_val_loss), 1):
        diff = val - train
        if diff < 0.30:
            status = "✅ 良好"
        elif diff < 0.35:
            status = "⚠️ 警戒"
        else:
            status = "🚨 过拟合"
            
        print(f"     {i}   |  {train:.6f} |  {val:.6f} |  {diff:.3f}  |  {status}")
    
    print("\n   过拟合临界点: 第4轮之后")
    print("   建议: 使用第4轮的模型进行部署")

if __name__ == "__main__":
    # 打印训练总结
    print_training_summary()
    
    # 分析过拟合
    analyze_overfitting()
    
    # 绘制可视化图表
    plot_training_progress()
    
    print("\n🎉 训练进度分析完成！")
