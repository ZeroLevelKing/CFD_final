import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager

def set_chinese_font():
    """设置中文字体支持"""
    try:
        # 查找系统支持中文的字体
        system_fonts = font_manager.findSystemFonts()
        chinese_fonts = [f for f in system_fonts if any(lang in f.lower() for lang in ['simhei', 'simsun', 'microsoftyahei', 'kaiti', 'stkaiti', 'fangsong', 'stfangsong'])]
        
        if chinese_fonts:
            # 使用找到的第一个中文字体
            plt.rcParams['font.sans-serif'] = [os.path.basename(chinese_fonts[0]).split('.')[0]]
            print(f"使用中文字体: {plt.rcParams['font.sans-serif'][0]}")
        else:
            # 使用默认字体，但尝试支持中文
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            print("警告: 未找到系统中文字体，使用备用字体")
        
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"字体设置失败: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

def plot_solution_comparison(params, U_num, exact_data, t, title='数值解与精确解对比', filename=None):
    """
    绘制数值解与精确解的对比图
    
    参数:
    params -- 计算参数字典
    U_num -- 数值解的守恒变量数组 [ρ, ρu, E]
    exact_data -- 精确解数据字典 {'x': x_exact, 'rho': rho_exact, 'u': u_exact, 'p': p_exact}
    t -- 当前时间
    title -- 图表标题 (可选)
    filename -- 保存文件名 (可选，不指定则不保存)
    """
    set_chinese_font()  # 设置中文字体支持
    
    # 从守恒变量计算数值解的原始变量
    gamma = params['gamma']
    rho_num = U_num[0]
    u_num = U_num[1] / rho_num
    p_num = (gamma - 1) * (U_num[2] - 0.5 * rho_num * u_num**2)
    
    # 精确解数据
    x_exact = exact_data['x']
    rho_exact = exact_data['rho']
    u_exact = exact_data['u']
    p_exact = exact_data['p']
    
    # 创建图表
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 密度分布对比
    axs[0].plot(params['x'], rho_num, 'b-', linewidth=1.5, label='数值解')
    axs[0].plot(x_exact, rho_exact, 'r--', linewidth=1.5, label='精确解')
    axs[0].set_ylabel(r'$\rho$', fontsize=12)
    axs[0].set_title(f'密度分布 (t={t:.2f})', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()
    
    # 速度分布对比
    axs[1].plot(params['x'], u_num, 'g-', linewidth=1.5, label='数值解')
    axs[1].plot(x_exact, u_exact, 'm--', linewidth=1.5, label='精确解')
    axs[1].set_ylabel(r'$u$', fontsize=12)
    axs[1].set_title(f'速度分布 (t={t:.2f})', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()
    
    # 压强分布对比
    axs[2].plot(params['x'], p_num, 'c-', linewidth=1.5, label='数值解')
    axs[2].plot(x_exact, p_exact, 'y--', linewidth=1.5, label='精确解')
    axs[2].set_ylabel(r'$p$', fontsize=12)
    axs[2].set_xlabel(r'$x$', fontsize=12)
    axs[2].set_title(f'压强分布 (t={t:.2f})', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()
    
    # 主标题
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    # 保存或显示图像
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"已保存对比图: {filename}")
        plt.close()
    else:
        plt.show()

def plot_error_analysis(params, errors, nx_values, method_name):
    """
    绘制误差随网格分辨率变化的分析图
    
    参数:
    params -- 计算参数字典
    errors -- 误差字典 {'rho': [误差数组], 'u': [误差数组], 'p': [误差数组]}
    nx_values -- 网格分辨率数组
    method_name -- 数值方法名称
    """
    set_chinese_font()  # 设置中文字体支持
    
    plt.figure(figsize=(10, 8))
    
    # 密度误差收敛性
    plt.subplot(3, 1, 1)
    plt.loglog(nx_values, errors['rho'], 'bo-', markersize=8)
    plt.xlabel('网格点数 (log)', fontsize=12)
    plt.ylabel(r'密度 $L_2$ 误差 (log)', fontsize=12)
    plt.title(f'{method_name} 方法 - 密度误差收敛性', fontsize=14)
    plt.grid(True, which="both", ls="--")
    
    # 速度误差收敛性
    plt.subplot(3, 1, 2)
    plt.loglog(nx_values, errors['u'], 'go-', markersize=8)
    plt.xlabel('网格点数 (log)', fontsize=12)
    plt.ylabel(r'速度 $L_2$ 误差 (log)', fontsize=12)
    plt.title(f'{method_name} 方法 - 速度误差收敛性', fontsize=14)
    plt.grid(True, which="both", ls="--")
    
    # 压强误差收敛性
    plt.subplot(3, 1, 3)
    plt.loglog(nx_values, errors['p'], 'co-', markersize=8)
    plt.xlabel('网格点数 (log)', fontsize=12)
    plt.ylabel(r'压强 $L_2$ 误差 (log)', fontsize=12)
    plt.title(f'{method_name} 方法 - 压强误差收敛性', fontsize=14)
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.savefig(f'{method_name}_误差收敛性.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_method_comparison(comparison_data, t):
    """
    绘制不同数值方法的对比图
    
    参数:
    comparison_data -- 方法对比数据字典，格式:
        {
            'TVD+FVS': {'x': x, 'rho': rho_tvd, 'u': u_tvd, 'p': p_tvd},
            'WENO+Roe': {'x': x, 'rho': rho_weno, 'u': u_weno, 'p': p_weno},
            ...
        }
    t -- 当前时间
    """
    set_chinese_font()  # 设置中文字体支持
    
    # 创建图表
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    markers = ['-', '--', '-.', ':']
    
    # 密度分布对比
    for i, (method, data) in enumerate(comparison_data.items()):
        axs[0].plot(data['x'], data['rho'], linestyle=markers[i%4], 
                   color=colors[i%7], linewidth=1.5, label=method)
    axs[0].set_ylabel(r'$\rho$', fontsize=12)
    axs[0].set_title(f'不同方法密度分布对比 (t={t:.2f})', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()
    
    # 速度分布对比
    for i, (method, data) in enumerate(comparison_data.items()):
        axs[1].plot(data['x'], data['u'], linestyle=markers[i%4], 
                   color=colors[i%7], linewidth=1.5, label=method)
    axs[1].set_ylabel(r'$u$', fontsize=12)
    axs[1].set_title(f'不同方法速度分布对比 (t={t:.2f})', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()
    
    # 压强分布对比
    for i, (method, data) in enumerate(comparison_data.items()):
        axs[2].plot(data['x'], data['p'], linestyle=markers[i%4], 
                   color=colors[i%7], linewidth=1.5, label=method)
    axs[2].set_ylabel(r'$p$', fontsize=12)
    axs[2].set_xlabel(r'$x$', fontsize=12)
    axs[2].set_title(f'不同方法压强分布对比 (t={t:.2f})', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()
    
    # 主标题
    fig.suptitle('数值方法性能对比', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    plt.savefig(f'数值方法对比_t={t:.2f}.png', dpi=300, bbox_inches='tight')
    plt.close()