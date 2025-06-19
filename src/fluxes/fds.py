# src/fluxes/fds.py

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
    
    # 计算左状态原始变量（添加多重保护）
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / max(rho_L, 1e-10)
    e_L = max(U_L[2] - 0.5 * rho_L * min(u_L**2, 1e10), 1e-10)
    p_L = max((gamma - 1) * e_L, 1e-10)
    
    # 计算右状态原始变量
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / max(rho_R, 1e-10)
    e_R = max(U_R[2] - 0.5 * rho_R * min(u_R**2, 1e10), 1e-10)
    p_R = max((gamma - 1) * e_R, 1e-10)
    
    # 计算左右通量（添加保护）
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
    
    # 计算中间状态的原始变量（添加保护）
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


def roe_flux(states, gamma):
    """
    Roe近似黎曼解算器 (增强数值稳定性)
    
    参数:
    states -- 左右状态数组 [ρ, ρu, E], 形状为 (3, 2)
    gamma -- 比热比
    
    返回:
    通量向量 (3,)
    """
    # 提取左右状态
    U_L = states[:, 0]
    U_R = states[:, 1]
    
    # 计算左状态原始变量（添加多重保护）
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / max(rho_L, 1e-10)
    e_L = max(U_L[2] - 0.5 * rho_L * min(u_L**2, 1e10), 1e-10)
    p_L = max((gamma - 1) * e_L, 1e-10)
    H_L = (U_L[2] + p_L) / max(rho_L, 1e-10)
    
    # 计算右状态原始变量（添加多重保护）
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / max(rho_R, 1e-10)
    e_R = max(U_R[2] - 0.5 * rho_R * min(u_R**2, 1e10), 1e-10)
    p_R = max((gamma - 1) * e_R, 1e-10)
    H_R = (U_R[2] + p_R) / max(rho_R, 1e-10)
    
    # 计算Roe平均
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    sum_sqrt = max(sqrt_rho_L + sqrt_rho_R, 1e-10)  # 防止除以零
    
    # Roe平均密度
    rho_roe = sqrt_rho_L * sqrt_rho_R
    
    # Roe平均速度
    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / sum_sqrt
    
    # Roe平均总焓
    H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / sum_sqrt
    
    # Roe平均声速（添加保护）
    H_minus_u2 = max(H_roe - 0.5 * min(u_roe**2, 1e10), 1e-10)
    a_roe = np.sqrt(max((gamma - 1) * H_minus_u2, 1e-10))
    
    # 计算左右通量
    F_L = np.array([
        rho_L * u_L,
        rho_L * min(u_L**2, 1e10) + p_L,  # 限制过大速度
        u_L * min(U_L[2] + p_L, 1e10)     # 限制过大能量
    ])
    
    F_R = np.array([
        rho_R * u_R,
        rho_R * min(u_R**2, 1e10) + p_R,  # 限制过大速度
        u_R * min(U_R[2] + p_R, 1e10)     # 限制过大能量
    ])
    
    # 计算特征值
    lambda1 = u_roe
    lambda2 = u_roe + a_roe
    lambda3 = u_roe - a_roe
    
    # 计算特征向量
    delta_U = U_R - U_L
    
    # 计算波浪强度
    delta_p = p_R - p_L
    delta_u = u_R - u_L
    
    # Roe差分（添加保护）
    denom = max(2 * a_roe**2, 1e-10)
    k = np.zeros(3)
    k[0] = (delta_p - rho_roe * a_roe * delta_u) / denom
    k[1] = (delta_p + rho_roe * a_roe * delta_u) / denom
    k[2] = (delta_U[0] - delta_p / max(a_roe**2, 1e-10))
    
    # Roe差分向量
    delta_F1 = k[0] * np.abs(lambda1) * np.array([1, u_roe, 0.5*min(u_roe**2, 1e10)])
    delta_F2 = k[1] * np.abs(lambda2) * np.array([1, u_roe + a_roe, min(H_roe + u_roe*a_roe, 1e10)])
    delta_F3 = k[2] * np.abs(lambda3) * np.array([1, u_roe - a_roe, min(H_roe - u_roe*a_roe, 1e10)])
    
    # Roe通量
    F = 0.5 * (F_L + F_R) - 0.5 * (delta_F1 + delta_F2 + delta_F3)
    
    # 最终保护：确保通量值合理
    F = np.nan_to_num(F, nan=0.0, posinf=1e10, neginf=-1e10)
    return np.clip(F, -1e10, 1e10)  # 限制通量范围