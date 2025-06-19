import numpy as np
import matplotlib.pyplot as plt
import os
from config import params
from initialization.domain_setup import create_domain
from initialization.sod_initial import initialize_sod, set_initial_conditions
from utils.boundary import apply_boundary_conditions, add_ghost_cells, remove_ghost_cells
from utils.exact_solution import compute_exact_solution, interpolate_exact_to_grid
from utils.visualization import plot_solution_comparison
from time_integration import rk3, compute_dt
from fluxes.fvs import get_flux_function as get_fvs_flux

def compute_characteristic_matrices(u, c, gamma):
    """
    计算特征矩阵 (左矩阵L和右矩阵R)
    
    参数:
    u -- 速度
    c -- 声速
    gamma -- 比热比
    
    返回:
    右特征矩阵 R, 左特征矩阵 L
    """
    # 添加数值保护
    u = np.clip(u, -1e5, 1e5)
    c = max(c, 1e-10)
    
    # 计算总焓
    H = c**2 / (gamma - 1) + 0.5 * u**2
    
    # 右特征矩阵
    R = np.array([
        [1, 1, 1],
        [u - c, u, u + c],
        [H - u * c, 0.5 * u**2, H + u * c]
    ])
    
    # 左特征矩阵 (R的逆)
    inv_factor = 1 / (c**2)
    beta = (gamma - 1) / (2 * c**2)
    
    L = np.array([
        [0.5*(beta*(u**2) + u/c), -0.5*(beta*2*u + 1/c), 0.5*beta],
        [1 - beta*u**2, beta*2*u, -beta],
        [0.5*(beta*(u**2) - u/c), -0.5*(beta*2*u - 1/c), 0.5*beta]
    ])
    
    return R, L

def weno5_z_reconstruction(v):
    """
    五阶WENO-Z重构 (适用于小模板)
    
    参数:
    v -- 包含5个相邻单元值的数组 [v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}]
    
    返回:
    重构值 v_rec
    """
    # 理想权重
    gamma = np.array([0.1, 0.6, 0.3])
    
    # 三个子模板的重构值
    v0 = (2*v[0] - 7*v[1] + 11*v[2]) / 6.0
    v1 = (-v[1] + 5*v[2] + 2*v[3]) / 6.0
    v2 = (2*v[2] + 5*v[3] - v[4]) / 6.0
    
    # 计算光滑指示器 (添加数值保护)
    try:
        beta0 = 13/12*(v[0] - 2*v[1] + v[2])**2 + 1/4*(v[0] - 4*v[1] + 3*v[2])**2
        beta1 = 13/12*(v[1] - 2*v[2] + v[3])**2 + 1/4*(v[1] - v[3])**2
        beta2 = 13/12*(v[2] - 2*v[3] + v[4])**2 + 1/4*(3*v[2] - 4*v[3] + v[4])**2
    except:
        # 如果出现数值问题，使用简单平均
        return np.mean(v)
    
    # 添加数值保护
    beta0 = max(beta0, 1e-10)
    beta1 = max(beta1, 1e-10)
    beta2 = max(beta2, 1e-10)
    
    # 计算全局光滑指示器
    tau5 = abs(beta0 - beta2)
    
    # 避免除零
    epsilon = 1e-6
    
    # 计算权重 (WENO-Z公式)
    alpha0 = gamma[0] * (1.0 + tau5/(beta0 + epsilon))
    alpha1 = gamma[1] * (1.0 + tau5/(beta1 + epsilon))
    alpha2 = gamma[2] * (1.0 + tau5/(beta2 + epsilon))
    
    alpha_sum = alpha0 + alpha1 + alpha2
    
    # 防止除零
    if alpha_sum < 1e-10:
        w0, w1, w2 = gamma
    else:
        w0 = alpha0 / alpha_sum
        w1 = alpha1 / alpha_sum
        w2 = alpha2 / alpha_sum
    
    # 加权平均
    return w0*v0 + w1*v1 + w2*v2

def minmod(a, b):
    """
    Minmod 限制器函数
    """
    if a * b <= 0:
        return 0
    else:
        return np.sign(a) * min(abs(a), abs(b))

def tvd_reconstruction(v):
    """
    TVD重构 (适用于小模板)
    
    参数:
    v -- 包含3个相邻单元值的数组 [v_{i-1}, v_i, v_{i+1}]
    
    返回:
    重构值 v_rec
    """
    # 应用限制器
    phi = minmod(v[1] - v[0], v[2] - v[1])
    
    # 重构值 (中心值)
    return v[1] - 0.5 * phi

def characteristic_reconstruction(U, gamma, reconstructor_type='weno', num_ghost=3):
    """
    特征空间重构
    
    参数:
    U -- 守恒变量 [ρ, ρu, E], 形状 (3, nx)
    gamma -- 比热比
    reconstructor_type -- 重构器类型 ('weno' 或 'tvd')
    num_ghost -- 虚单元层数
    
    返回:
    重构后的状态 U_char, 形状 (3, nx)
    """
    nx = U.shape[1]
    U_char = np.zeros_like(U)
    
    # 对每个单元计算特征矩阵
    for i in range(num_ghost, nx - num_ghost):
        # 计算单元状态 (添加数值保护)
        rho = max(U[0, i], 1e-10)
        u_val = U[1, i] / rho
        u_val = np.clip(u_val, -1e5, 1e5)  # 限制速度范围
        
        e = max(U[2, i] - 0.5 * rho * u_val**2, 1e-10)
        p = max((gamma - 1) * e, 1e-10)
        c = max(np.sqrt(gamma * p / rho), 1e-10)
        
        # 计算特征矩阵
        try:
            R, L = compute_characteristic_matrices(u_val, c, gamma)
        except:
            # 如果计算特征矩阵失败，使用原始值
            U_char[:, i] = U[:, i]
            continue
            
        # 应用重构器
        if reconstructor_type == 'weno':
            # 提取WENO模板 (5个点)
            stencil_idx = [i-2, i-1, i, i+1, i+2]
            if min(stencil_idx) < 0 or max(stencil_idx) >= nx:
                # 边界处理: 使用一阶重构
                U_char[:, i] = U[:, i]
                continue
                
            # 提取状态模板
            stencil = U[:, stencil_idx]
            
            # 对每个特征变量进行重构
            w_char_rec = np.zeros(3)
            for j in range(3):  # 对每个特征变量
                # 转换到特征空间
                w_char = np.zeros(5)
                for k in range(5):
                    try:
                        w_char[k] = L[j, :] @ stencil[:, k]
                    except:
                        w_char[k] = stencil[j, k]  # 如果失败，使用原始值
                
                # WENO重构 (添加异常处理)
                try:
                    w_char_rec[j] = weno5_z_reconstruction(w_char)
                except:
                    w_char_rec[j] = stencil[j, 2]  # 使用中心值
            
            # 转换回守恒空间
            try:
                U_char[:, i] = R @ w_char_rec
            except:
                U_char[:, i] = U[:, i]  # 如果失败，使用原始值
        else:
            # 提取TVD模板 (3个点)
            stencil_idx = [i-1, i, i+1]
            if min(stencil_idx) < 0 or max(stencil_idx) >= nx:
                # 边界处理: 使用一阶重构
                U_char[:, i] = U[:, i]
                continue
                
            # 提取状态模板
            stencil = U[:, stencil_idx]
            
            # 对每个特征变量进行重构
            w_char_rec = np.zeros(3)
            for j in range(3):  # 对每个特征变量
                # 转换到特征空间
                w_char = np.zeros(3)
                for k in range(3):
                    try:
                        w_char[k] = L[j, :] @ stencil[:, k]
                    except:
                        w_char[k] = stencil[j, k]  # 如果失败，使用原始值
                
                # TVD重构 (添加异常处理)
                try:
                    w_char_rec[j] = tvd_reconstruction(w_char)
                except:
                    w_char_rec[j] = stencil[j, 1]  # 使用中心值
            
            # 转换回守恒空间
            try:
                U_char[:, i] = R @ w_char_rec
            except:
                U_char[:, i] = U[:, i]  # 如果失败，使用原始值
                
        # 确保重构后的状态是物理的
        U_char[0, i] = max(U_char[0, i], 1e-10)  # 密度
        U_char[2, i] = max(U_char[2, i], 1e-10)  # 能量
    
    return U_char

def run_characteristic_simulation():
    """运行特征重构FVS的独立仿真"""
    # 创建输出目录
    os.makedirs("characteristic_results", exist_ok=True)
    
    # 创建计算域
    domain = create_domain(
        nx=params.nx, 
        x_min=params.x_min, 
        x_max=params.x_max, 
        t_end=params.t_end
    )
    
    # 设置初始条件
    physics = initialize_sod(domain, gamma=params.gamma, cv=params.cv)
    domain_params = set_initial_conditions(physics, domain)
    
    # 初始化守恒变量
    U = domain_params['U_init'].copy()
    
    # 获取通量函数
    flux_func = get_fvs_flux('van_leer')
    
    # 主循环
    t = 0.0
    step = 0
    next_output = 0.2  # 输出间隔
    
    print("运行特征重构FVS仿真...")
    print(f"网格数: {params.nx}, 时间终点: {params.t_end}")
    print(f"使用重构器: weno")
    
    while t < params.t_end:
        # 计算时间步长
        dt = compute_dt(U, domain_params['dx'], params.cfl, params.gamma)
        dt = min(dt, params.t_end - t)
        
        # 应用边界条件
        U_bc = apply_boundary_conditions(U, domain_params, num_ghost=params.num_ghost)
        
        # 添加虚单元
        U_ghost = add_ghost_cells(U_bc, num_ghost=params.num_ghost)
        
        # 应用特征重构
        U_char = characteristic_reconstruction(
            U_ghost, params.gamma, reconstructor_type='weno', num_ghost=params.num_ghost
        )
        
        # 计算通量
        nx = U_ghost.shape[1]
        F = np.zeros((3, nx - 1))
        
        # 计算每个界面的通量
        for i in range(nx - 1):
            # 获取界面两侧状态 (使用重构后的状态)
            state_left = U_char[:, i]
            state_right = U_char[:, i+1]
            states = np.array([state_left, state_right]).T
            
            # 计算通量 (添加异常处理)
            try:
                F[:, i] = flux_func(states, params.gamma)
            except:
                # 如果通量计算失败，使用简单平均
                rho_avg = 0.5 * (state_left[0] + state_right[0])
                u_avg = 0.5 * (state_left[1]/max(state_left[0],1e-10) + state_right[1]/max(state_right[0],1e-10))
                p_avg = 0.5 * ((params.gamma-1)*(state_left[2]-0.5*state_left[0]*(state_left[1]/max(state_left[0],1e-10))**2) + 
                             (params.gamma-1)*(state_right[2]-0.5*state_right[0]*(state_right[1]/max(state_right[0],1e-10))**2))
                F[:, i] = np.array([
                    rho_avg * u_avg,
                    rho_avg * u_avg**2 + p_avg,
                    u_avg * (0.5*rho_avg*u_avg**2 + p_avg/(params.gamma-1) + p_avg)
                ])
        
        # 计算通量散度 (dF/dx)
        dFdx = np.zeros_like(U_ghost)
        
        # 内部点: (F_{i+1/2} - F_{i-1/2})/dx
        for i in range(1, nx - 1):
            dFdx[:, i] = (F[:, i] - F[:, i-1]) / domain_params['dx']
        
        # 边界点使用一阶近似
        dFdx[:, 0] = dFdx[:, 1]
        dFdx[:, -1] = dFdx[:, -2]
        
        # 移除虚单元
        dFdx = remove_ghost_cells(dFdx, num_ghost=params.num_ghost)
        
        # 时间推进
        U_new = rk3(U, lambda U: -dFdx, dt, domain_params)
        
        # 更新变量
        U = U_new
        
        # 确保物理性
        U[0] = np.maximum(U[0], 1e-10)  # 密度
        U[2] = np.maximum(U[2], 1e-10)  # 能量
        
        # 更新时间和步数
        t += dt
        step += 1
        
        # 定期输出
        if t >= next_output:
            print(f"时间: {t:.4f}/{params.t_end:.2f}, 步数: {step}")
            next_output += 0.2
    
    # 最终结果处理和可视化
    exact_data = compute_exact_solution(domain_params, params.t_end)
    exact_on_grid = interpolate_exact_to_grid(domain_params, exact_data)
    
    # 保存结果
    plot_file = os.path.join(
        "results", 
        f"characteristic_fvs_t={params.t_end}.png"
    )
    plot_solution_comparison(
        domain_params, U, exact_on_grid, params.t_end,
        title=f"特征重构FVS (nx={params.nx}, t={params.t_end})",
        filename=plot_file
    )
    
    print(f"特征重构FVS仿真完成! 结果保存至: {plot_file}")

if __name__ == "__main__":
    # 设置参数
    params.nx = 200
    params.t_end = 1.0
    params.cfl = 0.3  # 降低CFL数以提高稳定性
    params.num_ghost = 3
    
    run_characteristic_simulation()