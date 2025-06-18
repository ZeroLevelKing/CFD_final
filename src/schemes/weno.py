import numpy as np

def lax_friedrichs_split(U, flux_func, gamma):
    """
    Lax-Friedrichs 通量分裂
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    flux_func -- 通量计算函数
    gamma -- 比热比
    
    返回:
    F_p, F_n -- 正负通量分量
    """
    # 计算原始变量
    rho = np.maximum(U[0], 1e-10)
    u = U[1] / rho
    e = np.maximum(U[2] - 0.5 * rho * u**2, 1e-10)
    p = np.maximum((gamma - 1) * e, 1e-10)
    
    # 计算声速和最大特征速度
    c = np.sqrt(gamma * p / rho)
    alpha = np.max(np.abs(u) + c)
    
    # 计算通量
    F = np.zeros_like(U)
    for i in range(U.shape[1]):
        states = U[:, i:i+1].T
        F[:, i] = flux_func(states, gamma)
    
    # 通量分裂
    F_p = 0.5 * (F + alpha * U)
    F_n = 0.5 * (F - alpha * U)
    
    return F_p, F_n

def weno5_fs(F, axis=0):
    """
    5阶WENO重构算法 (基于通量分裂)
    
    参数:
    F -- 通量分量 (正或负), 形状为 (3, nx)
    axis -- 重构方向 (0: 按变量; 1: 按网格点)
    
    返回:
    Fh -- 重构后的半网格点通量, 形状为 (3, nx)
    """
    # 确保F是二维数组
    if len(F.shape) == 1:
        F = F.reshape(1, -1)
    
    n_vars, n_points = F.shape
    
    # 初始化输出
    Fh = np.zeros((n_vars, n_points))
    
    # 理想权重
    C = np.array([1/10, 6/10, 3/10])
    p = 2
    eps = 1e-6
    
    # 起始和结束索引 (避免边界)
    xs = 2  # 起始索引
    xt = n_points - 3  # 结束索引
    
    for var in range(n_vars):
        F_var = F[var, :]
        
        for j in range(xs, xt + 1):
            # 计算光滑指示器 beta
            beta1 = (1/4)*(F_var[j-2] - 4*F_var[j-1] + 3*F_var[j])**2 + (13/12)*(F_var[j-2] - 2*F_var[j-1] + F_var[j])**2
            beta2 = (1/4)*(F_var[j-1] - F_var[j+1])**2 + (13/12)*(F_var[j-1] - 2*F_var[j] + F_var[j+1])**2
            beta3 = (1/4)*(3*F_var[j] - 4*F_var[j+1] + F_var[j+2])**2 + (13/12)*(F_var[j] - 2*F_var[j+1] + F_var[j+2])**2
            
            # 计算 alpha 和权重 omega
            alpha = C / (eps + np.array([beta1, beta2, beta3]))**p
            omega = alpha / np.sum(alpha)
            
            # 计算三个模板的通量值
            F_c1 = (1/3)*F_var[j-2] - (7/6)*F_var[j-1] + (11/6)*F_var[j]
            F_c2 = (-1/6)*F_var[j-1] + (5/6)*F_var[j] + (1/3)*F_var[j+1]
            F_c3 = (1/3)*F_var[j] + (5/6)*F_var[j+1] - (1/6)*F_var[j+2]
            
            # 加权组合得到 F_{j+1/2}
            Fh[var, j] = omega[0]*F_c1 + omega[1]*F_c2 + omega[2]*F_c3
    
    return Fh

def weno_fs_flux(U, flux_func, gamma, params):
    """
    基于通量分裂的WENO格式通量计算
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    flux_func -- 通量计算函数
    gamma -- 比热比
    params -- 计算参数字典
    
    返回:
    F -- 通量数组, 形状为 (3, nx-1)
    """
    nx = U.shape[1]
    F = np.zeros((3, nx-1))
    
    # 应用通量分裂
    F_p, F_n = lax_friedrichs_split(U, flux_func, gamma)
    
    # 正通量分量重构 (F+)
    Fh_p = weno5_fs(F_p)
    
    # 负通量分量重构 (F-)
    Fh_n = weno5_fs(F_n)
    
    # 组合通量
    for j in range(1, nx-2):  # 避开边界
        # 正通量在 j+1/2 处
        # 负通量在 j-1/2 处
        F[:, j] = Fh_p[:, j] + Fh_n[:, j+1]
    
    # 边界处理 (一阶)
    F[:, 0] = F_p[:, 0] + F_n[:, 1]
    F[:, nx-2] = F_p[:, nx-2] + F_n[:, nx-1]
    
    return F

def weno_fs_rhs(U, flux_func, gamma, params):
    """
    基于通量分裂的WENO格式右端项计算
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    flux_func -- 通量计算函数
    gamma -- 比热比
    params -- 计算参数字典
    
    返回:
    RHS -- 空间离散项, 形状为 (3, nx)
    """
    dx = params['dx']
    nx = U.shape[1]
    
    # 计算通量
    F = weno_fs_flux(U, flux_func, gamma, params)
    
    # 计算通量差
    RHS = np.zeros_like(U)
    
    # 内部点
    for i in range(1, nx-1):
        RHS[:, i] = - (F[:, i] - F[:, i-1]) / dx
    
    # 边界点 (一阶)
    RHS[:, 0] = - (F[:, 0] - np.zeros(3)) / dx
    RHS[:, nx-1] = - (np.zeros(3) - F[:, nx-2]) / dx
    
    return RHS