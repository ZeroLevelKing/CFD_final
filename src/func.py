import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

def initialize(nx=200, t_end=2.0, cfl=0.5):
    """
    初始化计算参数和网格
    
    参数:
    nx -- 网格点数 (默认200)
    t_end -- 计算结束时间 (默认2.0)
    cfl -- CFL稳定性条件数 (默认0.5)
    
    返回:
    包含所有计算参数的字典
    """
    # 物理常数
    gamma = 1.4  # 比热比 (空气)
    cv = 1.0     # 定容比热容 (无量纲化)
    
    # 计算域设置
    x_min = -5.0
    x_max = 5.0
    
    # 网格参数
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min, x_max, nx)  # 网格中心坐标
    
    # 初始条件 (Sod问题)
    rho = np.where(x < 0, 1.0, 0.125)  # 密度
    u = np.zeros_like(x)               # 速度
    p = np.where(x < 0, 1.0, 0.1)      # 压强
    
    # 守恒变量初始化 [ρ, ρu, E]
    E = p / (gamma - 1) + 0.5 * rho * u**2
    U = np.array([rho, rho * u, E])
    
    # 时间步参数
    dt = 0.0  # 将由CFL条件计算
    
    # 边界条件类型
    bc_type = 'non-reflective'  # 无反射边界
    
    # 收集所有参数
    params = {
        'nx': nx,
        'dx': dx,
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        't_end': t_end,
        'cfl': cfl,
        'U_init': U,
        'gamma': gamma,
        'cv': cv,
        'dt': dt,
        'bc_type': bc_type
    }
    
    return params

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

def plot_solution(params, U, t, title='解分布', filename=None):
    """
    绘制任意时刻的速度、压强、密度分布
    
    参数:
    params -- 计算参数字典
    U -- 守恒变量数组 [ρ, ρu, E]
    t -- 当前时间
    title -- 图表标题 (可选)
    filename -- 保存文件名 (可选，不指定则不保存)
    """

    set_chinese_font()  # 设置中文字体支持

    x = params['x']
    gamma = params['gamma']
    
    # 从守恒变量计算原始变量
    rho = U[0]
    u = U[1] / rho
    p = (gamma - 1) * (U[2] - 0.5 * rho * u**2)
    
    # 创建图表
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 密度分布
    axs[0].plot(x, rho, 'b-', linewidth=1.5, label='数值解')
    axs[0].set_ylabel(r'$\rho$')
    axs[0].set_title(f'密度分布 (t={t:.2f})')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()
    
    # 速度分布
    axs[1].plot(x, u, 'r-', linewidth=1.5, label='数值解')
    axs[1].set_ylabel(r'$u$')
    axs[1].set_title(f'速度分布 (t={t:.2f})')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()
    
    # 压强分布
    axs[2].plot(x, p, 'g-', linewidth=1.5, label='数值解')
    axs[2].set_ylabel(r'$p$')
    axs[2].set_xlabel(r'$x$')
    axs[2].set_title(f'压强分布 (t={t:.2f})')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()
    
    # 主标题
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # 保存或显示图像
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"已保存解分布图: {filename}")
    else:
        plt.show()
    
    plt.close()

    
    