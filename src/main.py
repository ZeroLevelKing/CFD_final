# src/main.py

import os
import time
import numpy as np
from config import params
from initialization.domain_setup import create_domain
from initialization.sod_initial import initialize_sod, set_initial_conditions
from utils.boundary import apply_boundary_conditions, add_ghost_cells, remove_ghost_cells
from utils.exact_solution import compute_exact_solution, interpolate_exact_to_grid, calculate_error
from utils.visualization import plot_solution_comparison
from time_integration import rk3, compute_dt

# 导入FVS通量函数
from fluxes.fvs import get_flux_function as get_fvs_flux

# 导入FDS通量函数
from fluxes.fds import get_fds_flux_function

def get_flux_function():
    """根据配置获取通量函数"""
    if params.flux_method == 'fvs':
        return get_fvs_flux(flux_type=params.flux_type)
    else:  # FDS
        return get_fds_flux_function(flux_type=params.flux_type)

def get_scheme_function():
    """根据配置获取激波捕捉格式函数"""
    if params.scheme == 'tvd':
        # 返回一个可以接受额外参数的函数
        def tvd_wrapper(U, flux_func, **kwargs):
            from schemes.tvd import tvd_flux
            return tvd_flux(U, flux_func, gamma=params.gamma, limiter=params.limiter, **kwargs)
        return tvd_wrapper
        
    elif params.scheme == 'gvc':
        # 返回一个可以接受额外参数的函数
        def gvc_wrapper(U, flux_func, **kwargs):
            from schemes.gvc import gvc_flux
            return gvc_flux(U, flux_func, gamma=params.gamma, **kwargs)
        return gvc_wrapper
        
    elif params.scheme == 'weno':
        # 返回一个可以接受额外参数的函数
        def weno_wrapper(U, flux_func, **kwargs):
            from schemes.weno import weno_flux
            return weno_flux(
                U, flux_func, gamma=params.gamma, 
                num_ghost=params.num_ghost, variant=params.weno_variant, **kwargs)
        return weno_wrapper
        
    else:
        raise ValueError(f"未知的激波捕捉格式: {params.scheme}")

def compute_rhs(U, flux_func, scheme_func, dx, dt):
    """计算空间离散项 (dU/dt)"""
    # 应用边界条件
    U_bc = apply_boundary_conditions(U, domain_params, num_ghost=params.num_ghost)
    
    # 添加虚单元
    U_ghost = add_ghost_cells(U_bc, num_ghost=params.num_ghost)
    
    # 计算通量 - 传递所有必要的参数
    if params.flux_type == 'lax_wendroff':
        # Lax-Wendroff 需要额外的 dx 和 dt 参数
        F = scheme_func(U_ghost, flux_func, dx=dx, dt=dt)
    else:
        # 其他通量函数
        F = scheme_func(U_ghost, flux_func)
    
    # 计算通量散度 (dF/dx)
    dFdx = np.zeros_like(U_ghost)
    n = U_ghost.shape[1]
    
    # 内部点: (F_{i+1/2} - F_{i-1/2})/dx
    for i in range(params.num_ghost, n - params.num_ghost):
        dFdx[:, i] = (F[:, i] - F[:, i-1]) / dx
    
    # 移除虚单元
    dFdx = remove_ghost_cells(dFdx, num_ghost=params.num_ghost)
    
    return -dFdx  # dU/dt = -dF/dx

def main():
    global domain_params
    # 创建输出目录
    os.makedirs(params.plot_dir, exist_ok=True)
    
    # 创建计算域
    domain_params = create_domain(
        nx=params.nx, 
        x_min=params.x_min, 
        x_max=params.x_max, 
        t_end=params.t_end
    )
    
    # 设置初始条件
    physics = initialize_sod(domain_params, gamma=params.gamma, cv=params.cv)
    domain_params = set_initial_conditions(physics, domain_params)
    
    # 初始化守恒变量
    U = domain_params['U_init'].copy()
    
    # 获取数值方法函数
    flux_func = get_flux_function()
    scheme_func = get_scheme_function()
    
    # 主循环
    t = 0.0
    step = 0
    next_output_time = 0.0
    
    print(f"开始计算: scheme={params.scheme}, flux={params.flux_type}")
    print(f"网格数: {params.nx}, 时间终点: {params.t_end}")
    
    while t < params.t_end:
        # 计算时间步长
        dt = compute_dt(U, domain_params['dx'], params.cfl, params.gamma)
        dt = min(dt, params.t_end - t)
        
        # 计算空间离散项 - 传递当前的 dx 和 dt
        rhs = compute_rhs(U, flux_func, scheme_func, domain_params['dx'], dt)
        
        # 时间推进 (三阶Runge-Kutta)
        U = rk3(U, lambda U: compute_rhs(U, flux_func, scheme_func, domain_params['dx'], dt), dt, domain_params)
        
        # 更新时间和步数
        t += dt
        step += 1
        
        # 输出进度
        if t >= next_output_time:
            print(f"时间: {t:.4f}/{params.t_end:.2f}, 步数: {step}, 时间步长: {dt:.2e}")
            next_output_time += params.output_interval
            
            # 计算并保存结果
            if params.save_plots:
                exact_data = compute_exact_solution(domain_params, t)
                exact_on_grid = interpolate_exact_to_grid(domain_params, exact_data)
                
                plot_file = os.path.join(
                    params.plot_dir, 
                    f"{params.scheme}_{params.flux_type}_t={t:.2f}.png"
                )
                plot_solution_comparison(
                    domain_params, U, exact_on_grid, t, 
                    title=f"{params.scheme} + {params.flux_type} (nx={params.nx})",
                    filename=plot_file
                )
    
    # 最终结果输出
    exact_data = compute_exact_solution(domain_params, params.t_end)
    exact_on_grid = interpolate_exact_to_grid(domain_params, exact_data)
    
    # 计算误差
    errors = calculate_error(U, exact_on_grid, domain_params)
    print("\n计算完成! 误差统计:")
    print(f"密度 L2 误差: {errors['rho']:.4e}")
    print(f"速度 L2 误差: {errors['u']:.4e}")
    print(f"压强 L2 误差: {errors['p']:.4e}")
    
    # 保存最终结果
    final_plot = os.path.join(
        params.plot_dir, 
        f"FINAL_{params.scheme}_{params.flux_type}_nx{params.nx}.png"
    )
    plot_solution_comparison(
        domain_params, U, exact_on_grid, params.t_end,
        title=f"{params.scheme} + {params.flux_type} (nx={params.nx}, t={params.t_end})",
        filename=final_plot
    )
    
    print(f"\n结果已保存到 {params.plot_dir} 目录")

if __name__ == "__main__":
    main()