# src/schemes/weno.py

import numpy as np

def weno5_js_reconstruction(v):
    """
    五阶WENO-JS重构 (Jiang & Shu, 1996)
    
    参数:
    v -- 包含5个相邻单元值的数组 [v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}]
    
    返回:
    界面i+1/2处的重构值
    """
    # 检查输入长度
    if len(v) != 5:
        raise ValueError(f"WENO重构需要5个点，但得到{len(v)}个点")
    
    # 理想权重
    gamma = np.array([0.1, 0.6, 0.3])
    
    # 三个子模板的重构值
    v0 = (2*v[0] - 7*v[1] + 11*v[2]) / 6.0
    v1 = (-v[1] + 5*v[2] + 2*v[3]) / 6.0
    v2 = (2*v[2] + 5*v[3] - v[4]) / 6.0
    
    # 光滑指示器 (smoothness indicators)
    beta0 = 13/12*(v[0] - 2*v[1] + v[2])**2 + 1/4*(v[0] - 4*v[1] + 3*v[2])**2
    beta1 = 13/12*(v[1] - 2*v[2] + v[3])**2 + 1/4*(v[1] - v[3])**2
    beta2 = 13/12*(v[2] - 2*v[3] + v[4])**2 + 1/4*(3*v[2] - 4*v[3] + v[4])**2
    
    # 避免除零
    epsilon = 1e-6
    
    # 计算权重
    alpha0 = gamma[0] / (epsilon + beta0)**2
    alpha1 = gamma[1] / (epsilon + beta1)**2
    alpha2 = gamma[2] / (epsilon + beta2)**2
    
    alpha_sum = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    
    # 加权平均
    return w0*v0 + w1*v1 + w2*v2

def weno5_z_reconstruction(v):
    """
    五阶WENO-Z重构 
    
    参数:
    v -- 包含5个相邻单元值的数组 [v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}]
    
    返回:
    界面i+1/2处的重构值
    """
    # 检查输入长度
    if len(v) != 5:
        raise ValueError(f"WENO重构需要5个点，但得到{len(v)}个点")
    
    # 理想权重
    gamma = np.array([0.1, 0.6, 0.3])
    
    # 三个子模板的重构值
    v0 = (2*v[0] - 7*v[1] + 11*v[2]) / 6.0
    v1 = (-v[1] + 5*v[2] + 2*v[3]) / 6.0
    v2 = (2*v[2] + 5*v[3] - v[4]) / 6.0
    
    # 光滑指示器 (smoothness indicators)
    beta0 = 13/12*(v[0] - 2*v[1] + v[2])**2 + 1/4*(v[0] - 4*v[1] + 3*v[2])**2
    beta1 = 13/12*(v[1] - 2*v[2] + v[3])**2 + 1/4*(v[1] - v[3])**2
    beta2 = 13/12*(v[2] - 2*v[3] + v[4])**2 + 1/4*(3*v[2] - 4*v[3] + v[4])**2
    
    # 计算全局光滑指示器
    tau5 = np.abs(beta0 - beta2)
    
    # 避免除零
    epsilon = 1e-6
    
    # 计算权重 (WENO-Z公式)
    alpha0 = gamma[0] * (1.0 + (tau5 / (epsilon + beta0)))
    alpha1 = gamma[1] * (1.0 + (tau5 / (epsilon + beta1)))
    alpha2 = gamma[2] * (1.0 + (tau5 / (epsilon + beta2)))
    
    alpha_sum = alpha0 + alpha1 + alpha2
    
    w0 = alpha0 / alpha_sum
    w1 = alpha1 / alpha_sum
    w2 = alpha2 / alpha_sum
    
    # 加权平均
    return w0*v0 + w1*v1 + w2*v2

def weno_reconstruction(U, num_ghost=3, variant='z'):
    """
    对守恒变量进行WENO重构
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, n)
    num_ghost -- 虚单元层数 (至少3)
    variant -- WENO变体 ('js' 或 'z')
    
    返回:
    重构后的左右状态 U_L, U_R, 形状均为 (3, n-1)
    """
    n = U.shape[1]
    n_interface = n - 1  # 界面数量
    
    # 检查虚单元层数
    if num_ghost < 3:
        raise ValueError("WENO重构需要至少3层虚单元")
    
    # 选择WENO变体
    if variant == 'js':
        weno_func = weno5_js_reconstruction
    elif variant == 'z':
        weno_func = weno5_z_reconstruction
    else:
        raise ValueError(f"未知的WENO变体: {variant}")
    
    # 初始化左右状态数组
    U_L = np.zeros((3, n_interface))
    U_R = np.zeros((3, n_interface))
    
    # 对每个变量分别处理
    for var in range(3):
        # 对每个界面进行重构（仅对内部点）
        for i in range(num_ghost, n - num_ghost - 1):
            # 左状态重构: 使用 [i-2, i-1, i, i+1, i+2]
            stencil_left = U[var, i-2:i+3]
            U_L[var, i] = weno_func(stencil_left)
            
            # 右状态重构: 使用 [i+3, i+2, i+1, i, i-1] 的逆序
            stencil_right = U[var, i-1:i+4][::-1]
            U_R[var, i] = weno_func(stencil_right)
    
    # 边界处理：使用一阶重构
    for var in range(3):
        for i in range(num_ghost):
            # 左边界
            U_L[var, i] = U[var, i+1]  # 右侧值
            U_R[var, i] = U[var, i+1]  # 右侧值
            
            # 右边界
            j = n_interface - 1 - i
            U_L[var, j] = U[var, j]    # 左侧值
            U_R[var, j] = U[var, j]    # 左侧值
    
    return U_L, U_R

def weno_flux(U, flux_func, gamma=None, num_ghost=3, variant='z', **kwargs):
    """
    WENO格式通量计算
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    flux_func -- 通量计算函数，接受左右状态作为参数
    gamma -- 比热比（可选）
    num_ghost -- 虚单元层数 (默认3)
    variant -- WENO变体 ('js' 或 'z', 默认'z')
    kwargs -- 传递给通量函数的额外参数
    
    返回:
    通量数组 F, 形状为 (3, nx-1)
    """
    # WENO重构
    U_L, U_R = weno_reconstruction(U, num_ghost, variant)
    
    nx = U.shape[1]
    n_interface = nx - 1
    F = np.zeros((3, n_interface))  # 界面通量比网格点数少1
    
    # 计算每个界面的通量
    for i in range(n_interface):
        # 使用重构后的左右状态计算通量
        state_left = U_L[:, i]
        state_right = U_R[:, i]
        
        # 创建状态数组 (2个状态点)
        states = np.array([state_left, state_right]).T
        
        # 计算通量
        if gamma is not None:
            flux_i = flux_func(states, gamma, **kwargs)
        else:
            flux_i = flux_func(states, **kwargs)
        
        # 存储通量
        F[:, i] = flux_i
    
    return F