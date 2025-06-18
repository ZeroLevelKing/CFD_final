import numpy as np
import matplotlib.pyplot as plt
from schemes.weno import weno_fs_rhs
from fluxes.fvs import get_flux_function
from time_integration import rk3, compute_dt
from initialization.sod_initial import initialize_sod, set_initial_conditions
from initialization.domain_setup import create_domain
from utils.boundary import apply_boundary_conditions, add_ghost_cells, remove_ghost_cells
from utils.exact_solution import compute_exact_solution, interpolate_exact_to_grid
from utils.visualization import plot_solution_comparison

# 设置计算参数
nx = 200
x_min = -5.0
x_max = 5.0
t_end = 0.2
gamma = 1.4
cfl = 0.5
num_ghost = 3

# 创建计算域
domain = create_domain(nx, x_min, x_max, t_end)
physics = initialize_sod(domain, gamma)
params = set_initial_conditions(physics, domain)
params['num_ghost'] = num_ghost
params['cfl'] = cfl

# 添加虚单元
U = add_ghost_cells(params['U_init'], num_ghost)

# 获取通量函数
flux_func = get_flux_function('steger_warming')

# 时间推进
t = 0.0
step = 0
max_steps = 10000

while t < t_end and step < max_steps:
    # 应用边界条件
    U = apply_boundary_conditions(U, params, num_ghost)
    
    # 计算时间步长
    dt = min(compute_dt(U, params['dx'], cfl, gamma), t_end - t)
    
    # 移除虚单元计算RHS
    U_internal = remove_ghost_cells(U, num_ghost)
    RHS = weno_fs_rhs(U_internal, flux_func, gamma, params)
    
    # 添加虚单元用于时间推进
    RHS_extended = add_ghost_cells(RHS, num_ghost)
    
    # RK3时间推进
    U = rk3(U, lambda U: RHS_extended, dt, params)
    
    # 更新时间和步数
    t += dt
    step += 1
    
    # 打印进度
    if step % 100 == 0:
        print(f"Step {step}, Time = {t:.4f}, dt = {dt:.4f}")

# 移除虚单元
U_final = remove_ghost_cells(U, num_ghost)

# 计算精确解
exact_data = compute_exact_solution(params, t)
exact_interp = interpolate_exact_to_grid(params, exact_data)

# 绘制比较图
plot_solution_comparison(params, U_final, exact_interp, t, 
                         title='WENO-FS + Steger-Warming 数值解与精确解对比',
                         filename='weno_fs_steger_comparison.png')