import numpy as np

def minmod(a, b):
    """
    Minmod 限制器函数 (支持数组输入)
    
    参数:
    a, b -- 输入值 (标量或数组)
    
    返回:
    minmod 值 (与输入同形状)
    """
    # 处理数组输入
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        # 创建结果数组
        result = np.zeros_like(a)
        
        # 处理每个元素
        for i in range(len(a)):
            if a[i] * b[i] <= 0:
                result[i] = 0
            else:
                result[i] = np.sign(a[i]) * min(abs(a[i]), abs(b[i]))
        
        return result
    else:
        # 处理标量输入
        if a * b <= 0:
            return 0
        else:
            return np.sign(a) * min(abs(a), abs(b))


def muscl_reconstruction(U, limiter='minmod'):
    """
    MUSCL (Monotone Upstream-centered Scheme for Conservation Laws) 重构
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    limiter -- 限制器类型 ('minmod', 'superbee', 'van_leer')
    
    返回:
    重构后的左右状态 U_L, U_R, 形状均为 (3, nx)
    """
    nx = U.shape[1]
    U_L = np.zeros_like(U)
    U_R = np.zeros_like(U)
    
    # 对每个变量分别处理
    for var in range(3):
        # 计算梯度
        left_grad = U[var, 1:] - U[var, :-1]  # i 到 i+1
        right_grad = U[var, 2:] - U[var, 1:-1]  # i+1 到 i+2
        
        # 需要调整数组大小以匹配
        phi = np.zeros(nx-2)
        
        # 应用限制器
        for i in range(nx-2):
            if limiter == 'minmod':
                phi[i] = minmod(left_grad[i], right_grad[i])
            else:
                phi[i] = 0  # 无限制器
        
        # 重构左右状态 (内部点)
        U_L[var, 1:-1] = U[var, 1:-1] + 0.5 * phi
        U_R[var, 1:-1] = U[var, 1:-1] - 0.5 * phi
    
    # 边界处理
    U_L[:, 0] = U[:, 0]
    U_R[:, 0] = U[:, 0]
    U_L[:, -1] = U[:, -1]
    U_R[:, -1] = U[:, -1]
    
    return U_L, U_R


def tvd_flux(U, flux_func, gamma=None, limiter='minmod', **kwargs):
    """
    TVD 格式通量计算
    
    参数:
    U -- 守恒变量数组 [ρ, ρu, E], 形状为 (3, nx)
    flux_func -- 通量计算函数，接受左右状态作为参数
    gamma -- 比热比（可选）
    limiter -- 限制器类型 ('minmod', 'superbee', 'van_leer')
    kwargs -- 传递给通量函数的额外参数
    
    返回:
    通量数组 F, 形状为 (3, nx-1)
    """
    # MUSCL 重构
    U_L, U_R = muscl_reconstruction(U, limiter)
    
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