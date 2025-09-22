#!/usr/bin/env python3
"""
Timeloop映射策略可视化分析工具
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_ws_mapping():
    """可视化Weight Stationary映射的数据访问模式"""
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # 时间步和数据范围
    timesteps = ['t/0/', 't/1/', 't/2/']
    
    # Weight访问模式
    weight_ranges = [[0, 1], [1, 2], [2, 3]]
    ax1.barh(timesteps, [1, 1, 1], left=[0, 1, 2], 
             color=['red', 'green', 'blue'], alpha=0.7)
    ax1.set_xlabel('Weight Index')
    ax1.set_title('Weight访问模式 - 时间复用')
    ax1.set_xlim(-0.5, 3.5)
    
    # Input访问模式 - 滑动窗口
    input_starts = [0, 1, 2]
    input_widths = [16, 16, 16]
    for i, (start, width) in enumerate(zip(input_starts, input_widths)):
        ax2.barh(timesteps[i], width, left=start, 
                color=f'C{i}', alpha=0.7, label=f'Input[{start}:{start+width})')
    ax2.set_xlabel('Input Index')
    ax2.set_title('Input访问模式 - 滑动窗口')
    ax2.set_xlim(-0.5, 18.5)
    ax2.legend()
    
    # Output访问模式 - 所有时间步访问相同范围
    for i, timestep in enumerate(timesteps):
        ax3.barh(timestep, 16, left=0, 
                color='orange', alpha=0.7)
    ax3.set_xlabel('Output Index')
    ax3.set_title('Output访问模式 - 累加存储')
    ax3.set_xlim(-0.5, 16.5)
    
    plt.tight_layout()
    plt.savefig('ws_mapping_pattern.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_storage_requirements():
    """比较不同映射策略的存储需求"""
    
    strategies = ['Weight\nStationary', 'Output\nStationary']
    
    # 存储需求数据 (以元素数量计)
    weight_storage = [1, 3]  # WS每次1个，OS需要完整3个
    input_storage = [16, 3]  # WS需要滑动窗口16个，OS只需当前3个  
    output_storage = [16, 1] # WS需要完整16个累加，OS只需当前1个
    
    x = np.arange(len(strategies))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, weight_storage, width, label='Weight', color='red', alpha=0.7)
    rects2 = ax.bar(x, input_storage, width, label='Input', color='green', alpha=0.7)
    rects3 = ax.bar(x + width, output_storage, width, label='Output', color='blue', alpha=0.7)
    
    ax.set_ylabel('存储元素数量')
    ax.set_title('映射策略存储需求对比')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    
    # 在柱状图上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig('storage_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def energy_analysis():
    """分析不同映射策略的能耗特征"""
    
    # 基于Timeloop输出: pJ/Compute = 7.241
    ws_energy_per_compute = 7.241
    
    # 能耗分解估算 (基于典型的存储器能耗模型)
    compute_energy = 0.845  # MAC操作能耗
    buffer_access_energy = 0.36  # Buffer访问能耗  
    memory_access_energy = 2.0   # 主存访问能耗
    
    # WS映射的能耗分解
    ws_breakdown = {
        'Compute (MAC)': compute_energy,
        'Buffer Access': buffer_access_energy * 3,  # Weight+Input+Output
        'Memory Access': memory_access_energy * 2,  # 假设2次主存访问
        'Others': ws_energy_per_compute - compute_energy - buffer_access_energy * 3 - memory_access_energy * 2
    }
    
    # 创建饼图
    fig, ax = plt.subplots(figsize=(8, 8))
    
    labels = list(ws_breakdown.keys())
    sizes = list(ws_breakdown.values())
    colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen']
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 10})
    
    ax.set_title(f'Weight Stationary映射能耗分解\n总计: {ws_energy_per_compute:.3f} pJ/Compute', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ws_energy_breakdown.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("生成Weight Stationary映射分析图表...")
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        visualize_ws_mapping()
        print("✅ 数据访问模式图表已生成")
        
        compare_storage_requirements()
        print("✅ 存储需求对比图表已生成")
        
        energy_analysis()
        print("✅ 能耗分解图表已生成")
        
    except Exception as e:
        print(f"❌ 图表生成失败: {e}")
        print("提示: 如果是字体问题，可以注释掉中文字体设置行")
