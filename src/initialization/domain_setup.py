import numpy as np

def create_domain(nx=200, x_min=-5.0, x_max=5.0, t_end=2.0):
    """
    创建计算域和网格
    
    参数:
    nx -- 网格点数 (默认200)
    x_min -- 计算域左边界 (默认-5.0)
    x_max -- 计算域右边界 (默认5.0)
    t_end -- 计算结束时间 (默认2.0)
    
    返回:
    包含所有网格参数的字典
    """
    # 计算网格参数
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min, x_max, nx)  # 网格中心坐标
    
    # 收集所有参数
    domain = {
        'nx': nx,
        'dx': dx,
        'x': x,
        'x_min': x_min,
        'x_max': x_max,
        't_end': t_end,
        'dt': 0.0,  # 将由CFL条件计算
    }
    
    return domain