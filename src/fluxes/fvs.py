# src/fluxes/fvs.py

import numpy as np

def steger_warming_flux(states, gamma):
    """
    Steger-Warming 通量矢量分裂 (修正版)
    
    参数:
    states -- 左右状态数组 [ρ, ρu, E], 形状为 (3, 2)
    gamma -- 比热比
    
    返回:
    通量向量 (3,)
    
    参考: Steger, J. L., & Warming, R. F. (1981). 
        Flux vector splitting of the inviscid gasdynamic equations with application to finite-difference methods.
        Journal of Computational Physics, 40(2), 263-293.
    """
    # 提取左右状态
    U_L = states[:, 0]
    U_R = states[:, 1]
    
    # 计算左状态原始变量
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    p_L = max((gamma - 1) * (U_L[2] - 0.5 * rho_L * u_L**2), 1e-10)
    c_L = np.sqrt(gamma * p_L / rho_L)
    H_L = (U_L[2] + p_L) / rho_L
    
    # 计算右状态原始变量
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    p_R = max((gamma - 1) * (U_R[2] - 0.5 * rho_R * u_R**2), 1e-10)
    c_R = np.sqrt(gamma * p_R / rho_R)
    H_R = (U_R[2] + p_R) / rho_R
    
    # 计算左状态的特征值
    lambda1_L = u_L
    lambda2_L = u_L + c_L
    lambda3_L = u_L - c_L
    
    # 分裂特征值 (左状态)
    lambda1_plus_L = 0.5 * (lambda1_L + np.abs(lambda1_L))
    lambda2_plus_L = 0.5 * (lambda2_L + np.abs(lambda2_L))
    lambda3_plus_L = 0.5 * (lambda3_L + np.abs(lambda3_L))
    
    # 计算左状态的分裂通量 F⁺
    F_plus = np.zeros(3)
    F_plus[0] = rho_L/(2*gamma) * (2*(gamma-1)*lambda1_plus_L + lambda2_plus_L + lambda3_plus_L)
    F_plus[1] = rho_L/(2*gamma) * (2*(gamma-1)*lambda1_plus_L*u_L + 
                                  lambda2_plus_L*(u_L + c_L) + 
                                  lambda3_plus_L*(u_L - c_L))
    F_plus[2] = rho_L/(2*gamma) * ((gamma-1)*lambda1_plus_L*u_L**2 + 
                                  0.5*lambda2_plus_L*(u_L + c_L)**2 + 
                                  0.5*lambda3_plus_L*(u_L - c_L)**2 +
                                  (3-gamma)/(2*(gamma-1)) * (lambda2_plus_L + lambda3_plus_L - 2*(gamma-1)*lambda1_plus_L)*c_L**2)
    
    # 计算右状态的特征值
    lambda1_R = u_R
    lambda2_R = u_R + c_R
    lambda3_R = u_R - c_R
    
    # 分裂特征值 (右状态)
    lambda1_minus_R = 0.5 * (lambda1_R - np.abs(lambda1_R))
    lambda2_minus_R = 0.5 * (lambda2_R - np.abs(lambda2_R))
    lambda3_minus_R = 0.5 * (lambda3_R - np.abs(lambda3_R))
    
    # 计算右状态的分裂通量 F⁻
    F_minus = np.zeros(3)
    F_minus[0] = rho_R/(2*gamma) * (2*(gamma-1)*lambda1_minus_R + lambda2_minus_R + lambda3_minus_R)
    F_minus[1] = rho_R/(2*gamma) * (2*(gamma-1)*lambda1_minus_R*u_R + 
                                   lambda2_minus_R*(u_R + c_R) + 
                                   lambda3_minus_R*(u_R - c_R))
    F_minus[2] = rho_R/(2*gamma) * ((gamma-1)*lambda1_minus_R*u_R**2 + 
                                   0.5*lambda2_minus_R*(u_R + c_R)**2 + 
                                   0.5*lambda3_minus_R*(u_R - c_R)**2 +
                                   (3-gamma)/(2*(gamma-1)) * (lambda2_minus_R + lambda3_minus_R - 2*(gamma-1)*lambda1_minus_R)*c_R**2)
    
    # 计算界面通量 F = F⁺(U_L) + F⁻(U_R)
    F = F_plus + F_minus
    
    return F

def van_leer_flux(states, gamma):
    """
    Van Leer 通量矢量分裂 (修正版)
    
    参数:
    states -- 左右状态数组 [ρ, ρu, E], 形状为 (3, 2)
    gamma -- 比热比
    
    返回:
    通量向量 (3,)
    
    参考: Van Leer, B. (1982). 
        Flux-vector splitting for the Euler equations. 
        In Lecture Notes in Physics (Vol. 170, pp. 507-512). Springer.
    """
    # 提取左右状态
    U_L = states[:, 0]
    U_R = states[:, 1]
    
    # 计算左状态原始变量
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    p_L = max((gamma - 1) * (U_L[2] - 0.5 * rho_L * u_L**2), 1e-10)
    c_L = np.sqrt(gamma * p_L / rho_L)
    
    # 计算右状态原始变量
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    p_R = max((gamma - 1) * (U_R[2] - 0.5 * rho_R * u_R**2), 1e-10)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # 计算马赫数
    M_L = u_L / c_L
    M_R = u_R / c_R
    
    # 初始化通量
    F = np.zeros(3)
    
    # 亚音速情况 (|M| < 1)
    if abs(M_L) < 1 and abs(M_R) < 1:
        # 左状态贡献
        f_mass_L = rho_L * c_L * (M_L + 1)**2 / 4
        f_mom_L = f_mass_L * (2 * c_L / gamma) * ((gamma - 1) * M_L / 2 + 1)
        f_energy_L = f_mass_L * (2 * c_L**2 / (gamma**2 - 1)) * ((gamma - 1) * M_L / 2 + 1)**2
        
        # 右状态贡献
        f_mass_R = -rho_R * c_R * (M_R - 1)**2 / 4
        f_mom_R = f_mass_R * (2 * c_R / gamma) * ((gamma - 1) * M_R / 2 - 1)
        f_energy_R = f_mass_R * (2 * c_R**2 / (gamma**2 - 1)) * ((gamma - 1) * M_R / 2 - 1)**2
        
        # 组合通量
        F[0] = f_mass_L + f_mass_R
        F[1] = f_mom_L + f_mom_R
        F[2] = f_energy_L + f_energy_R
    
    # 超音速情况
    else:
        # 完全超音速从左到右
        if M_L > 1 and M_R > 1:
            F[0] = rho_L * u_L
            F[1] = rho_L * u_L**2 + p_L
            F[2] = u_L * (U_L[2] + p_L)
        # 完全超音速从右到左
        elif M_L < -1 and M_R < -1:
            F[0] = rho_R * u_R
            F[1] = rho_R * u_R**2 + p_R
            F[2] = u_R * (U_R[2] + p_R)
        # 混合情况 (激波)
        else:
            # 使用更鲁棒的HLL通量作为后备
            return hll_flux(states, gamma)
    
    return F

def hll_flux(states, gamma):
    """
    HLL 近似黎曼解算器 (作为 Van Leer 的后备)
    
    参数:
    states -- 左右状态数组 [ρ, ρu, E], 形状为 (3, 2)
    gamma -- 比热比
    
    返回:
    通量向量 (3,)
    """
    # 提取左右状态
    U_L = states[:, 0]
    U_R = states[:, 1]
    
    # 计算左状态原始变量
    rho_L = max(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    p_L = max((gamma - 1) * (U_L[2] - 0.5 * rho_L * u_L**2), 1e-10)
    c_L = np.sqrt(gamma * p_L / rho_L)
    
    # 计算右状态原始变量
    rho_R = max(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    p_R = max((gamma - 1) * (U_R[2] - 0.5 * rho_R * u_R**2), 1e-10)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # 计算左右通量
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        u_L * (U_L[2] + p_L)
    ])
    
    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        u_R * (U_R[2] + p_R)
    ])
    
    # 估计波速
    S_L = min(u_L - c_L, u_R - c_R)
    S_R = max(u_L + c_L, u_R + c_R)
    
    # 计算HLL通量
    if S_L >= 0:
        return F_L
    elif S_R <= 0:
        return F_R
    else:
        return (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)

def ausm_flux(states, gamma):
    """
    AUSM (Advection Upstream Splitting Method) 通量 (适用于单个界面)
    
    参数:
    states -- 左右状态数组 [ρ, ρu, E], 形状为 (3, 2)
    gamma -- 比热比
    
    返回:
    通量向量 (3,)
    """
    # 提取左右状态
    U_L = states[:, 0]
    U_R = states[:, 1]
    
    # 计算左状态原始变量
    rho_L = U_L[0]
    u_L = U_L[1] / max(rho_L, 1e-10)
    p_L = (gamma - 1) * max(U_L[2] - 0.5 * rho_L * u_L**2, 1e-10)
    c_L = np.sqrt(gamma * p_L / max(rho_L, 1e-10))
    H_L = (U_L[2] + p_L) / max(rho_L, 1e-10)
    
    # 计算右状态原始变量
    rho_R = U_R[0]
    u_R = U_R[1] / max(rho_R, 1e-10)
    p_R = (gamma - 1) * max(U_R[2] - 0.5 * rho_R * u_R**2, 1e-10)
    c_R = np.sqrt(gamma * p_R / max(rho_R, 1e-10))
    H_R = (U_R[2] + p_R) / max(rho_R, 1e-10)
    
    # 平均声速
    c_avg = 0.5 * (c_L + c_R)
    
    # 马赫数分裂
    M_L = u_L / c_avg
    M_R = u_R / c_avg
    
    M_plus = 0.5 * (M_L + np.abs(M_L))
    M_minus = 0.5 * (M_R - np.abs(M_R))
    M_half = M_plus + M_minus
    
    # 压力分裂
    alpha = 3.0 / 16.0 * (-4.0 + 5.0 * gamma**2)
    beta = 1.0 / 8.0
    
    P_plus = 0.25 * p_L * (M_L + 1)**2 * (2 - M_L) + beta * M_L * (M_L**2 - 1)**2 * alpha * p_L
    P_minus = 0.25 * p_R * (M_R - 1)**2 * (2 + M_R) - beta * M_R * (M_R**2 - 1)**2 * alpha * p_R
    
    # 界面质量通量
    if M_half > 0:
        mass_flux = c_avg * M_half * rho_L
    else:
        mass_flux = c_avg * M_half * rho_R
    
    # 计算通量
    F = np.zeros(3)
    F[0] = mass_flux
    F[1] = mass_flux * (u_L if M_half > 0 else u_R) + P_plus + P_minus
    F[2] = mass_flux * (H_L if M_half > 0 else H_R)
    
    return F

def get_flux_function(flux_type='steger_warming'):
    """
    获取指定的通量函数
    
    参数:
    flux_type -- 通量类型 ('steger_warming', 'van_leer', 'ausm', 'lax_friedrichs')
    
    返回:
    通量计算函数
    """
    if flux_type == 'steger_warming':
        return steger_warming_flux
    elif flux_type == 'van_leer':
        return van_leer_flux
    elif flux_type == 'ausm':
        return ausm_flux
    elif flux_type == 'lax_friedrichs':
        return lax_friedrichs_flux
    else:
        raise ValueError(f"未知的通量类型: {flux_type}")

def lax_friedrichs_flux(states, gamma):
    """
    Lax-Friedrichs 通量 (适用于单个界面)
    
    参数:
    states -- 左右状态数组 [ρ, ρu, E], 形状为 (3, 2)
    gamma -- 比热比
    
    返回:
    通量向量 (3,)
    """
    # 提取左右状态
    U_L = states[:, 0]
    U_R = states[:, 1]
    
    # 计算左状态原始变量
    rho_L = U_L[0]
    u_L = U_L[1] / max(rho_L, 1e-10)
    p_L = (gamma - 1) * max(U_L[2] - 0.5 * rho_L * u_L**2, 1e-10)
    
    # 计算右状态原始变量
    rho_R = U_R[0]
    u_R = U_R[1] / max(rho_R, 1e-10)
    p_R = (gamma - 1) * max(U_R[2] - 0.5 * rho_R * u_R**2, 1e-10)
    
    # 计算左右通量
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        u_L * (U_L[2] + p_L)
    ])
    
    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        u_R * (U_R[2] + p_R)
    ])
    
    # 计算最大波速
    c_L = np.sqrt(gamma * p_L / max(rho_L, 1e-10))
    c_R = np.sqrt(gamma * p_R / max(rho_R, 1e-10))
    max_speed = max(np.abs(u_L) + c_L, np.abs(u_R) + c_R)
    
    # Lax-Friedrichs 通量
    F = 0.5 * (F_L + F_R) - 0.5 * max_speed * (U_R - U_L)
    
    return F


def central_flux(state, gamma):
    """
    计算单元中心通量 (欧拉方程的守恒通量)
    
    参数:
    state -- 状态向量 [ρ, ρu, E], 形状为 (3,)
    gamma -- 比热比
    
    返回:
    通量向量 (3,)
    """
    rho = max(state[0], 1e-10)
    u = state[1] / rho
    e = max(state[2] - 0.5 * rho * u**2, 1e-10)
    p = max((gamma - 1) * e, 1e-10)
    
    F = np.array([
        rho * u,
        rho * u**2 + p,
        u * (state[2] + p)
    ])
    
    return F

# 修改现有通量函数，添加一个包装器
def make_flux_function(flux_func):
    """
    创建通用通量函数，既能处理界面通量也能处理单元中心通量
    
    参数:
    flux_func -- 原始通量函数 (需要两个状态)
    
    返回:
    通用通量函数
    """
    def wrapper(states, gamma):
        if states.shape[1] == 1:
            # 单个状态 - 计算单元中心通量
            return central_flux(states[:, 0], gamma)
        elif states.shape[1] == 2:
            # 两个状态 - 计算界面通量
            return flux_func(states, gamma)
        else:
            raise ValueError("无效的状态数组形状")
    
    return wrapper

# 为每个通量函数创建包装器
steger_warming_flux_wrapped = make_flux_function(steger_warming_flux)
van_leer_flux_wrapped = make_flux_function(van_leer_flux)
ausm_flux_wrapped = make_flux_function(ausm_flux)
lax_friedrichs_flux_wrapped = make_flux_function(lax_friedrichs_flux)

def get_flux_function(flux_type='steger_warming'):
    """
    获取指定的通量函数 (包装后的版本)
    
    参数:
    flux_type -- 通量类型 ('steger_warming', 'van_leer', 'ausm', 'lax_friedrichs')
    
    返回:
    通量计算函数
    """
    if flux_type == 'steger_warming':
        return steger_warming_flux_wrapped
    elif flux_type == 'van_leer':
        return van_leer_flux_wrapped
    elif flux_type == 'ausm':
        return ausm_flux_wrapped
    elif flux_type == 'lax_friedrichs':
        return lax_friedrichs_flux_wrapped
    else:
        raise ValueError(f"未知的通量类型: {flux_type}")