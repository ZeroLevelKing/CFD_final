import numpy as np

def apply_boundary_conditions(U, params, num_ghost=2):
    """
    应用边界条件到守恒变量数组
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    params -- 计算参数字典
    num_ghost -- 虚单元层数 (默认2)
    
    返回:
    应用边界条件后的守恒变量数组
    """
    bc_type = params.get('bc_type', 'non-reflective')
    
    if bc_type == 'non-reflective':
        return non_reflective_bc(U, params, num_ghost)
    elif bc_type == 'periodic':
        return periodic_bc(U, num_ghost)
    elif bc_type == 'fixed':
        return fixed_bc(U, params, num_ghost)
    else:
        raise ValueError(f"未知边界条件类型: {bc_type}")

def non_reflective_bc(U, params, num_ghost):
    """
    无反射边界条件 (一阶外推)
    
    原理:
    为避免边界反射波，使用外推法设置边界值。
    对于左边界: U_{-i} = U_{0}
    对于右边界: U_{N+i} = U_{N-1}
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    params -- 计算参数字典
    num_ghost -- 虚单元层数
    
    返回:
    应用边界条件后的守恒变量数组
    """
    # 确保数组有足够的虚单元空间
    if U.shape[1] < num_ghost * 2:
        raise ValueError("数组大小不足以容纳指定数量的虚单元")
    
    # 应用左边界条件 (外推)
    for i in range(num_ghost):
        U[:, i] = U[:, num_ghost]
    
    # 应用右边界条件 (外推)
    n = U.shape[1]
    for i in range(num_ghost):
        U[:, n-1-i] = U[:, n-1-num_ghost]
    
    return U

def characteristic_bc(U, params, num_ghost):
    """
    特征边界条件 (更先进的无反射边界)
    
    原理:
    基于特征分析设置边界条件，更好地抑制数值反射。
    1. 计算特征变量
    2. 对进入计算域的特征变量外推
    3. 对离开计算域的特征变量保持不变
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    params -- 计算参数字典
    num_ghost -- 虚单元层数
    
    返回:
    应用边界条件后的守恒变量数组
    """
    gamma = params['gamma']
    
    # 确保数组有足够的虚单元空间
    if U.shape[1] < num_ghost * 2:
        raise ValueError("数组大小不足以容纳指定数量的虚单元")
    
    # 左边界处理
    # 计算特征变量
    rho, m, E = U[:, num_ghost]
    u = m / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    c = np.sqrt(gamma * p / rho)  # 声速
    
    # 特征变量: W = [u - 2c/(γ-1), u + 2c/(γ-1), s] (s为熵)
    W = np.array([
        u - 2*c/(gamma-1),
        u + 2*c/(gamma-1),
        p / rho**gamma  # 熵相关变量
    ])
    
    # 外推进入计算域的特征变量
    for i in range(num_ghost):
        # 只外推第一个特征变量 (u - 2c/(γ-1))
        W_extrap = W.copy()
        W_extrap[0] = W[0] - (i+1) * (W[0] - U[1, num_ghost+1] / U[0, num_ghost+1])
        
        # 转换回守恒变量
        u_bc = 0.5 * (W_extrap[1] + W_extrap[0])
        c_bc = 0.25 * (gamma-1) * (W_extrap[1] - W_extrap[0])
        rho_bc = (c_bc**2 / (gamma * W_extrap[2]))**(1/(gamma-1))
        p_bc = rho_bc * c_bc**2 / gamma
        E_bc = p_bc / (gamma-1) + 0.5 * rho_bc * u_bc**2
        
        U[:, i] = [rho_bc, rho_bc * u_bc, E_bc]
    
    # 右边界处理 (类似)
    # ...
    
    return U

def periodic_bc(U, num_ghost):
    """
    周期性边界条件
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    num_ghost -- 虚单元层数
    
    返回:
    应用边界条件后的守恒变量数组
    """
    n = U.shape[1]
    
    # 左边界: 复制右内部点
    U[:, :num_ghost] = U[:, n-2*num_ghost:n-num_ghost]
    
    # 右边界: 复制左内部点
    U[:, n-num_ghost:] = U[:, num_ghost:2*num_ghost]
    
    return U

def fixed_bc(U, params, num_ghost):
    """
    固定边界条件 (使用初始边界值)
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    params -- 计算参数字典
    num_ghost -- 虚单元层数
    
    返回:
    应用边界条件后的守恒变量数组
    """
    # 获取初始边界值
    rho_left = params['rho_init'][0]
    u_left = params['u_init'][0]
    p_left = params['p_init'][0]
    E_left = p_left / (params['gamma'] - 1) + 0.5 * rho_left * u_left**2
    
    rho_right = params['rho_init'][-1]
    u_right = params['u_init'][-1]
    p_right = params['p_init'][-1]
    E_right = p_right / (params['gamma'] - 1) + 0.5 * rho_right * u_right**2
    
    # 应用左边界条件
    for i in range(num_ghost):
        U[:, i] = [rho_left, rho_left * u_left, E_left]
    
    # 应用右边界条件
    n = U.shape[1]
    for i in range(num_ghost):
        U[:, n-1-i] = [rho_right, rho_right * u_right, E_right]
    
    return U

def add_ghost_cells(U, num_ghost):
    """
    为数组添加虚单元
    
    参数:
    U -- 原始守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    num_ghost -- 虚单元层数
    
    返回:
    扩展后的数组，形状为 (3, nx + 2*num_ghost)
    """
    n = U.shape[1]
    U_extended = np.zeros((3, n + 2 * num_ghost))
    
    # 复制内部点
    U_extended[:, num_ghost:num_ghost+n] = U
    
    return U_extended

def remove_ghost_cells(U, num_ghost):
    """
    移除数组的虚单元
    
    参数:
    U -- 带虚单元的守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    num_ghost -- 虚单元层数
    
    返回:
    移除虚单元后的数组，形状为 (3, nx - 2*num_ghost)
    """
    if U.shape[1] <= 2 * num_ghost:
        raise ValueError("数组太小，无法移除指定数量的虚单元")
    
    return U[:, num_ghost:-num_ghost]