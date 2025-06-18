# src/schemes/gvc.py

import numpy as np

def gvc_limiter(a, b):
    """
    GVC (群速度控制) 限制器函数 (支持数组输入)
    
    参数:
    a, b -- 输入值 (标量或数组)
    
    返回:
    GVC限制器值 (与输入同形状)
    
    原理:
    GVC限制器基于群速度控制的思想，旨在更好地处理高频振荡和非线性波传播问题。
    限制器公式：φ(r) = (2r)/(r^2 + 1)，其中 r = a/b
    """
    # 处理数组输入
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        # 创建结果数组
        result = np.zeros_like(a)
        
        # 处理每个元素
        for i in range(len(a)):
            if abs(b[i]) < 1e-10:  # 避免除以零
                result[i] = 0
            else:
                r = a[i] / b[i]
                # GVC限制器公式
                result[i] = (2 * r) / (r**2 + 1)
        
        return result
    else:
        # 处理标量输入
        if abs(b) < 1e-10:  # 避免除以零
            return 0
        else:
            r = a / b
            # GVC限制器公式
            return (2 * r) / (r**2 + 1)

def muscl_reconstruction_gvc(U, limiter='gvc'):
    """
    MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) 重构
    使用GVC限制器进行群速度控制
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    limiter -- 限制器类型 (这里专门为GVC设计)
    
    返回:
    重构后的左右状态 U_L, U_R, 形状均为 (3, nx)
    """
    nx = U.shape[1]
    U_L = np.zeros_like(U)
    U_R = np.zeros_like(U)
    
    # 对每个变量分别处理
    for var in range(3):
        # 计算梯度
        # 左侧梯度: i-1 到 i
        grad_left = U[var, 1:-1] - U[var, 0:-2]
        
        # 右侧梯度: i 到 i+1
        grad_right = U[var, 2:] - U[var, 1:-1]
        
        # 中心梯度: i-1 到 i+1 (用于GVC限制器)
        grad_center = U[var, 2:] - U[var, 0:-2]
        
        # 应用GVC限制器
        phi = gvc_limiter(grad_left, grad_center)
        
        # 重构左右状态 (内部点)
        U_L[var, 1:-1] = U[var, 1:-1] - 0.5 * phi * grad_left
        U_R[var, 1:-1] = U[var, 1:-1] + 0.5 * phi * grad_right
    
    # 边界处理 (一阶外推)
    U_L[:, 0] = U[:, 0]
    U_R[:, 0] = U[:, 0]
    U_L[:, -1] = U[:, -1]
    U_R[:, -1] = U[:, -1]
    
    return U_L, U_R

def gvc_flux(U, flux_func, gamma=None, **kwargs):
    """
    GVC (群速度控制) 格式通量计算
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    flux_func -- 通量计算函数，接受左右状态作为参数
    gamma -- 比热比（可选）
    kwargs -- 传递给通量函数的额外参数
    
    返回:
    通量数组 F, 形状为 (3, nx-1)
    """
    # GVC重构 - 使用专门设计的MUSCL重构
    U_L, U_R = muscl_reconstruction_gvc(U)
    
    nx = U.shape[1]
    F = np.zeros((3, nx-1))  # 界面通量比网格点数少1
    
    # 计算每个界面的通量
    for i in range(nx-1):
        # 使用重构后的左右状态计算通量
        # 第 i 个界面的左侧状态是 U_R[:, i]，右侧状态是 U_L[:, i+1]
        state_left = U_R[:, i]
        state_right = U_L[:, i+1]
        
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