import numpy as np
import matplotlib.pyplot as plt

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

