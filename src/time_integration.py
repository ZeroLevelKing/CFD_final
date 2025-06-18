import numpy as np


def rk3(U, RHS, dt, params):
    """
    三阶 Runge-Kutta 时间推进
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    RHS -- 空间离散项, 形状与 U 相同
    dt -- 时间步长
    params -- 计算参数字典，包含边界条件函数
    
    返回:
    更新后的守恒变量数组 U_new, 形状与 U 相同
    """
    # 获取边界条件函数
    boundary_func = params.get('boundary_func', lambda U, p: U)
    
    # 第一步
    U1 = U + dt * RHS(U)
    U1 = boundary_func(U1, params)  # 应用边界条件
    
    # 第二步
    U2 = 0.75 * U + 0.25 * U1 + 0.25 * dt * RHS(U1)
    U2 = boundary_func(U2, params)  # 应用边界条件
    
    # 第三步
    U_new = (1/3) * U + (2/3) * U2 + (2/3) * dt * RHS(U2)
    U_new = boundary_func(U_new, params)  # 应用边界条件
    
    return U_new



# 在 compute_dt 函数中添加更严格的保护

def compute_dt(U, dx, cfl, gamma):
    """
    根据CFL条件计算时间步长
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    dx -- 空间步长
    cfl -- CFL数
    gamma -- 比热比
    
    返回:
    时间步长 dt
    """
    # 计算原始变量（添加更严格的保护）
    rho = np.maximum(U[0], 1e-10)
    u = U[1] / np.maximum(rho, 1e-10)
    
    # 计算内能，确保非负
    e = np.maximum(U[2] - 0.5 * rho * np.minimum(u**2, 1e10), 1e-10)
    p = np.maximum((gamma - 1) * e, 1e-10)
    
    # 计算声速
    c = np.sqrt(gamma * p / np.maximum(rho, 1e-10))
    
    # 计算最大波速
    max_speed = np.max(np.abs(u) + c)
    
    # 计算时间步长
    if max_speed < 1e-10:
        # 避免除以零
        dt = cfl * dx
    else:
        dt = cfl * dx / max_speed
    
    # 添加最小时间步长限制
    min_dt = 1e-6
    dt = max(dt, min_dt)
    
    return dt