# src/fluxes/lax_wendroff.py

import numpy as np

def lax_wendroff_flux(states, gamma, dx, dt):
    """
    Lax-Wendroff 格式通量计算 (适用于单个界面)
    
    参数:
    states -- 左右状态数组 [ρ, ρu, E], 形状为 (3, 2)
    gamma -- 比热比
    dx -- 空间步长
    dt -- 时间步长
    
    返回:
    通量向量 (3,)
    """
    # 提取左右状态
    U_L = states[:, 0]
    U_R = states[:, 1]
    
    # 计算左状态原始变量（添加保护）
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / max(rho_L, 1e-10)
    e_L = max(U_L[2] - 0.5 * rho_L * min(u_L**2, 1e10), 1e-10)
    p_L = max((gamma - 1) * e_L, 1e-10)
    
    # 计算右状态原始变量
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / max(rho_R, 1e-10)
    e_R = max(U_R[2] - 0.5 * rho_R * min(u_R**2, 1e10), 1e-10)
    p_R = max((gamma - 1) * e_R, 1e-10)
    
    # 计算左右通量
    F_L = np.array([
        rho_L * u_L,
        rho_L * min(u_L**2, 1e10) + p_L,
        u_L * (U_L[2] + p_L)
    ])
    
    F_R = np.array([
        rho_R * u_R,
        rho_R * min(u_R**2, 1e10) + p_R,
        u_R * (U_R[2] + p_R)
    ])
    
    # 计算中间状态 U_{i+1/2}^{n+1/2}
    U_mid = 0.5 * (U_L + U_R) - 0.5 * (dt / dx) * (F_R - F_L)
    
    # 计算中间状态的原始变量
    rho_mid = max(U_mid[0], 1e-10)
    u_mid = U_mid[1] / max(rho_mid, 1e-10)
    e_mid = max(U_mid[2] - 0.5 * rho_mid * min(u_mid**2, 1e10), 1e-10)
    p_mid = max((gamma - 1) * e_mid, 1e-10)
    
    # 计算中间通量
    F_mid = np.array([
        rho_mid * u_mid,
        rho_mid * min(u_mid**2, 1e10) + p_mid,
        u_mid * (U_mid[2] + p_mid)
    ])
    
    # Lax-Wendroff 通量
    F = F_mid
    
    return F