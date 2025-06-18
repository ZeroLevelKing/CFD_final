# src/fds_test.py

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# 添加 src 目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from initialization.domain_setup import create_domain
from initialization.sod_initial import initialize_sod, set_initial_conditions
from utils.boundary import apply_boundary_conditions, add_ghost_cells, remove_ghost_cells
from utils.exact_solution import compute_exact_solution, interpolate_exact_to_grid
from utils.visualization import plot_solution_comparison, set_chinese_font
from schemes.tvd import tvd_flux
from fluxes.fds import lax_wendroff_flux  # 从fds模块导入Lax-Wendroff通量函数
from time_integration import rk3, compute_dt

def main():
    """主测试函数：Lax-Wendroff 通量 + TVD + Runge-Kutta"""
    print("="*70)
    print("Lax-Wendroff 通量 + TVD + Runge-Kutta 组合验证")
    print("="*70)
    
    # 设置中文字体
    set_chinese_font()
    
    # 创建测试结果目录
    os.makedirs("test_results/lw_flux", exist_ok=True)
    
    # 创建计算域和初始条件
    nx = 400
    domain = create_domain(nx=nx, x_min=-5.0, x_max=5.0, t_end=0.2)
    physics = initialize_sod(domain)
    params = set_initial_conditions(physics, domain)
    
    # 获取初始守恒变量
    U = params['U_init']
    
    # 添加虚单元
    num_ghost = 2
    U_extended = add_ghost_cells(U, num_ghost)
    
    # 设置边界条件函数
    def apply_bc(U, params):
        return apply_boundary_conditions(U, params, num_ghost)
    
    # 设置参数
    gamma = params['gamma']
    dx = params['dx']
    
    # 定义空间离散函数
    def compute_rhs(U):
        # 计算时间步长（使用当前状态）
        dt_local = compute_dt(U_extended, dx, cfl, gamma)
        
        # 计算通量
        F = tvd_flux(U, 
                    lax_wendroff_flux, 
                    gamma=gamma, 
                    limiter='minmod',
                    dx=dx,
                    dt=dt_local)
        
        # 计算空间导数
        RHS = np.zeros_like(U)
        RHS[:, 1:-1] = -(F[:, 1:] - F[:, :-1]) / dx
        
        return RHS
    
    # 设置时间参数
    t = 0.0
    t_end = params['t_end']
    cfl = 0.5  # 使用固定CFL数
    save_interval = 0.05  # 每0.05秒保存一次结果
    next_save_time = save_interval
    
    # 计算精确解 (初始时刻)
    exact_data = compute_exact_solution(params, t)
    exact_interp = interpolate_exact_to_grid(params, exact_data)
    
    # 初始可视化
    plot_filename = f"test_results/lw_flux/initial.png"
    plot_solution_comparison(params, remove_ghost_cells(U_extended, num_ghost), 
                            exact_interp, t, title='初始条件', filename=plot_filename)
    
    # 时间推进循环
    step = 0
    start_time = time.time()
    
    print(f"\n开始时间推进: t_end={t_end:.3f}, CFL={cfl}, nx={nx}")
    print(f"使用格式: Lax-Wendroff 通量 + TVD-minmod + RK3")
    print(f"每{save_interval:.2f}秒保存一次结果")
    
    # 调试信息
    debug_info = []
    
    # 添加详细调试输出
    def print_debug_info(U, step, t, dt):
        """打印详细的调试信息"""
        # 计算原始变量
        rho = U[0]
        u = U[1] / np.maximum(rho, 1e-10)
        p = (gamma - 1) * (U[2] - 0.5 * rho * np.minimum(u**2, 1e10))
        
        # 计算声速
        c = np.sqrt(gamma * p / np.maximum(rho, 1e-10))
        
        # 计算波速
        wave_speed = np.abs(u) + c
        
        print(f"步数: {step}, 极间: {t:.6f}, 步长: {dt:.6f}")
        print(f"密度范围: min={np.min(rho):.6f}, max={np.max(rho):.6f}")
        print(f"速度范围: min={np.min(u):.6f}, max={np.max(u):.6f}")
        print(f"压强范围: min={np.min(p):.6f}, max={np.max(p):.6f}")
        print(f"波速范围: min={np.min(wave_speed):.6f}, max={np.max(wave_speed):.6f}")
        print("-" * 60)
    
    # 初始调试信息
    print_debug_info(U_extended[:, num_ghost:-num_ghost], step, t, 0)
    
    while t < t_end:
        # 计算时间步长
        dt = compute_dt(U_extended, dx, cfl, gamma)
        
        # 确保不超过保存时间和结束时间
        if t + dt > next_save_time:
            dt = next_save_time - t
        if t + dt > t_end:
            dt = t_end - t
        
        # 应用边界条件
        U_extended = apply_bc(U_extended, params)
        
        # 检查边界条件后的状态
        if np.any(np.isnan(U_extended)):
            print("警告: 边界条件后出现NaN值!")
            print("左边界:", U_extended[:, :num_ghost])
            print("右边界:", U_extended[:, -num_ghost:])
            break
        
        try:
            # 时间推进
            U_extended = rk3(U_extended, compute_rhs, dt, {'boundary_func': apply_bc})
        except Exception as e:
            print(f"时间推进出错: {e}")
            # 保存当前状态用于调试
            np.save(f"test_results/lw_flux/error_state_step_{step}.npy", U_extended)
            
            # 打印详细状态信息
            print(f"当前时间: {t:.6f}, 步长: {dt:.6f}")
            print(f"U_extended min: {np.min(U_extended)}, max: {np.max(U_extended)}")
            
            # 尝试计算原始变量
            try:
                rho = U_extended[0]
                u = U_extended[1] / np.maximum(rho, 1e-10)
                p = (gamma - 1) * (U_extended[2] - 0.5 * rho * np.minimum(u**2, 1e10))
                print(f"密度范围: min={np.min(rho)}, max={np.max(rho)}")
                print(f"速度范围: min={np.min(u)}, max={np.max(u)}")
                print(f"压强范围: min={np.min(p)}, max={np.max(p)}")
            except:
                print("无法计算原始变量")
            
            # 终止程序
            raise
        
        # 更新时间
        t += dt
        step += 1
        
        # 每10步打印调试信息
        if step % 10 == 0:
            print_debug_info(U_extended[:, num_ghost:-num_ghost], step, t, dt)
        
        # 检查是否保存结果
        if t >= next_save_time - 1e-10:
            # 计算精确解
            exact_data = compute_exact_solution(params, t)
            exact_interp = interpolate_exact_to_grid(params, exact_data)
            
            # 移除虚单元
            U_internal = remove_ghost_cells(U_extended, num_ghost)
            
            # 可视化
            plot_filename = f"test_results/lw_flux/t={t:.3f}.png"
            plot_title = f"Lax-Wendroff通量+TVD+RK3 (t={t:.3f})"
            plot_solution_comparison(params, U_internal, exact_interp, t, 
                                    title=plot_title, filename=plot_filename)
            
            # 计算当前误差
            rho_num = U_internal[0]
            u_num = U_internal[1] / np.maximum(rho_num, 1e-10)
            p_num = (gamma - 1) * (U_internal[2] - 0.5 * rho_num * u_num**2)
            
            rho_exact = exact_interp['rho']
            u_exact = exact_interp['u']
            p_exact = exact_interp['p']
            
            error_rho = np.sqrt(np.mean((rho_num - rho_exact)**2))
            error_u = np.sqrt(np.mean((u_num - u_exact)**2))
            error_p = np.sqrt(np.mean((p_num - p_exact)**2))
            
            print(f"时间 t={t:.3f}, 步数={step}, 步长={dt:.3e}, 误差: ρ={error_rho:.4e}, u={error_u:.4e}, p={error_p:.4e}")
            
            # 保存调试信息
            debug_info.append({
                'step': step,
                't': t,
                'dt': dt,
                'error_rho': error_rho,
                'error_u': error_u,
                'error_p': error_p,
                'min_rho': np.min(rho_num),
                'max_rho': np.max(rho_num),
                'min_u': np.min(u_num),
                'max_u': np.max(u_num),
                'min_p': np.min(p_num),
                'max_p': np.max(p_num)
            })
            
            # 更新下次保存时间
            next_save_time += save_interval
    
    # 最终结果可视化
    plot_filename = f"test_results/lw_flux/final_t={t:.3f}.png"
    plot_title = f"Lax-Wendroff通量+TVD+RK3 (最终结果 t={t:.3f})"
    plot_solution_comparison(params, U_internal, exact_interp, t, 
                             title=plot_title, filename=plot_filename)
    
    # 计算性能
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n计算完成! 总步数: {step}, 总时间: {total_time:.2f}秒")
    print(f"平均每步时间: {total_time/step*1000:.2f}毫秒")
    
    # 保存调试信息
    import pandas as pd
    debug_df = pd.DataFrame(debug_info)
    debug_df.to_csv('test_results/lw_flux/debug_info.csv', index=False)
    
    print("\n测试完成! 结果保存至 test_results/lw_flux/ 目录")

if __name__ == "__main__":
    # 运行主测试
    main()
    
    print("\n所有测试完成!")