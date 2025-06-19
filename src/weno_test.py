# weno_z_sw_test.py

import numpy as np
import matplotlib.pyplot as plt
from initialization.sod_initial import initialize_sod
from initialization.domain_setup import create_domain
from time_integration import rk3, compute_dt
from fluxes.fvs import steger_warming_flux
from schemes.weno import weno_flux
from utils.boundary import apply_boundary_conditions, add_ghost_cells, remove_ghost_cells
from utils.exact_solution import compute_exact_solution
from utils.visualization import plot_solution_comparison
import time

def main():
    # 设置计算参数
    gamma = 1.4      # 比热比
    nx = 200         # 网格点数
    cfl = 0.4        # CFL数
    num_ghost = 3    # WENO需要3层虚单元
    t_end = 0.2      # 结束时间
    weno_variant = 'z'  # WENO-Z变体
    
    # 创建计算域
    domain = create_domain(nx=nx, x_min=-5.0, x_max=5.0, t_end=t_end)
    
    # 初始化Sod问题
    physics = initialize_sod(domain, gamma=gamma)
    
    # 合并参数
    params = {**domain, **physics}
    params['gamma'] = gamma
    params['cfl'] = cfl
    params['num_ghost'] = num_ghost
    
    # 添加虚单元
    U = add_ghost_cells(params['U_init'], num_ghost)
    
    # 选择通量函数 (FVS - Steger-Warming)
    flux_func = steger_warming_flux
    
    # 空间离散函数 (使用WENO-Z格式)
    def RHS(U_inner):
        # 应用边界条件
        U_bc = apply_boundary_conditions(U_inner, params, num_ghost)
        
        # 计算通量 (使用WENO-Z格式)
        F = weno_flux(U_bc, flux_func, gamma=gamma, num_ghost=num_ghost, variant=weno_variant)
        
        # 计算通量散度 (空间导数)
        dF = np.zeros_like(U_bc)
        n_total = U_bc.shape[1]
        
        # 只更新内部点的通量散度
        # 内部点索引: [num_ghost : n_total - num_ghost]
        for i in range(num_ghost, n_total - num_ghost):
            # 通量索引 = 单元索引 - num_ghost
            flux_index = i - num_ghost
            
            # 单元i的右侧通量: F[:, flux_index]
            # 单元i的左侧通量: F[:, flux_index - 1]
            dF[:, i] = (F[:, flux_index] - F[:, flux_index - 1]) / params['dx']
        
        return -dF  # 返回负通量散度（包含虚单元）
    
    # 时间推进
    t = 0.0
    iteration = 0
    start_time = time.time()
    
    # 存储结果用于绘图
    results = []
    times = [0.01, 0.02]  # 要保存结果的时间点
    
    while t < t_end:
        # 计算时间步长 - 使用内部点计算
        U_inner = remove_ghost_cells(U, num_ghost)
        dt = compute_dt(U_inner, params['dx'], cfl, gamma)
        dt = min(dt, t_end - t)  # 确保不超过结束时间
        
        # 时间推进 (RK3)
        U = rk3(U, RHS, dt, params)
        
        t += dt
        iteration += 1
        
        # 打印进度
        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            print(f"迭代: {iteration}, 时间: {t:.4f}, 时间步长: {dt:.6f}, 耗时: {elapsed:.2f}秒")
        
        # 保存特定时间点的结果
        if times and t >= times[0]:
            save_time = times.pop(0)
            # 移除虚单元并保存结果
            U_inner = remove_ghost_cells(U, num_ghost)
            results.append((save_time, U_inner.copy()))
            print(f"保存时间点 t={save_time} 的结果")
    
    # 如果没有达到保存时间点，保存最终结果
    if t_end not in [t for t, _ in results]:
        U_inner = remove_ghost_cells(U, num_ghost)
        results.append((t_end, U_inner))
    
    # 计算精确解并绘制比较图
    for t_save, U_num in results:
        # 计算精确解
        exact_data = compute_exact_solution(params, t_save)
        
        # 绘制比较图
        plot_solution_comparison(
            params, U_num, exact_data, t_save,
            title=f'WENO-Z+Steger-Warming求解Sod激波管 (t={t_save:.2f})',
            filename=f'weno_z_steger_comparison_t_{t_save:.2f}.png'
        )
    
    print(f"计算完成，总迭代步数: {iteration}, 总耗时: {time.time()-start_time:.2f}秒")

if __name__ == "__main__":
    main()