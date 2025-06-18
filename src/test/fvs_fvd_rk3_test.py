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
from fluxes.fvs import get_flux_function
from schemes.tvd import tvd_flux
from time_integration import rk3, compute_dt

def main():
    """主测试函数"""
    print("="*70)
    print("FVS + TVD + Runge-Kutta 组合验证")
    print("="*70)
    
    # 设置中文字体
    set_chinese_font()
    
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
    
    # 获取通量函数
    flux_func = get_flux_function('steger_warming')
    gamma = params['gamma']
    dx = params['dx']
    
    # 定义空间离散函数
    # fvs_fvd_rk3_test.py

    def compute_rhs(U):
        # 计算通量
        F = tvd_flux(U, flux_func, gamma, limiter='minmod')  # F 形状为 (3, nx-1)
        
        # 计算空间导数
        RHS = np.zeros_like(U)
        
        # 计算通量差: F_{i+1/2} - F_{i-1/2}
        # 注意: F 包含从 i=0 到 i=nx-2 的界面通量
        # 因此 F_{i+1/2} 对应 F[:, i], F_{i-1/2} 对应 F[:, i-1]
        # 内部点从 i=1 到 i=nx-2
        RHS[:, 1:-1] = -(F[:, 1:] - F[:, :-1]) / dx
        
        return RHS
    
    # 设置时间参数
    t = 0.0
    t_end = params['t_end']
    cfl = params['cfl']
    save_interval = 0.05  # 每0.05秒保存一次结果
    next_save_time = save_interval
    
    # 计算精确解 (初始时刻)
    exact_data = compute_exact_solution(params, t)
    exact_interp = interpolate_exact_to_grid(params, exact_data)
    
    # 初始可视化
    plot_filename = "test_results/tvd_fvs_initial.png"
    plot_solution_comparison(params, remove_ghost_cells(U_extended, num_ghost), 
                             exact_interp, t, title='初始条件', filename=plot_filename)
    
    # 时间推进循环
    step = 0
    start_time = time.time()
    
    print(f"\n开始时间推进: t_end={t_end:.3f}, CFL={cfl}, nx={nx}")
    print(f"每{save_interval:.2f}秒保存一次结果")
    
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
        
        # 时间推进
        U_extended = rk3(U_extended, compute_rhs, dt, {'boundary_func': apply_bc})
        
        # 更新时间
        t += dt
        step += 1
        
        # 检查是否保存结果
        if t >= next_save_time - 1e-10:
            # 计算精确解
            exact_data = compute_exact_solution(params, t)
            exact_interp = interpolate_exact_to_grid(params, exact_data)
            
            # 移除虚单元
            U_internal = remove_ghost_cells(U_extended, num_ghost)
            
            # 可视化
            plot_filename = f"test_results/tvd_fvs_t={t:.3f}.png"
            plot_title = f"FVS+TVD+RK3 (t={t:.3f})"
            plot_solution_comparison(params, U_internal, exact_interp, t, 
                                     title=plot_title, filename=plot_filename)
            
            # 计算当前误差
            rho_num = U_internal[0]
            u_num = U_internal[1] / rho_num
            p_num = (gamma - 1) * (U_internal[2] - 0.5 * rho_num * u_num**2)
            
            rho_exact = exact_interp['rho']
            u_exact = exact_interp['u']
            p_exact = exact_interp['p']
            
            error_rho = np.sqrt(np.mean((rho_num - rho_exact)**2))
            error_u = np.sqrt(np.mean((u_num - u_exact)**2))
            error_p = np.sqrt(np.mean((p_num - p_exact)**2))
            
            print(f"时间 t={t:.3f}, 步数={step}, 步长={dt:.3e}, 误差: ρ={error_rho:.4e}, u={error_u:.4e}, p={error_p:.4e}")
            
            # 更新下次保存时间
            next_save_time += save_interval
    
    # 最终结果可视化
    plot_filename = "test_results/tvd_fvs_final.png"
    plot_title = f"FVS+TVD+RK3 (最终结果 t={t:.3f})"
    plot_solution_comparison(params, U_internal, exact_interp, t, 
                             title=plot_title, filename=plot_filename)
    
    # 计算性能
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n计算完成! 总步数: {step}, 总时间: {total_time:.2f}秒")
    print(f"平均每步时间: {total_time/step*1000:.2f}毫秒")
    
    print("\n测试完成! 结果保存至 test_results/ 目录")

def test_minmod_limiter():
    """测试 Minmod 限制器"""
    # 测试标量输入
    from schemes.tvd import minmod
    
    # 测试标量
    print("\n测试 Minmod 限制器 (标量):")
    print("minmod(1.0, 0.5) =", minmod(1.0, 0.5), "预期: 0.5")
    print("minmod(-1.0, 0.5) =", minmod(-1.0, 0.5), "预期: 0")
    print("minmod(2.0, 1.5) =", minmod(2.0, 1.5), "预期: 1.5")
    print("minmod(0.5, -0.5) =", minmod(0.5, -0.5), "预期: 0")
    
    # 测试数组输入
    a = np.array([1.0, 2.0, -1.0, 0.5])
    b = np.array([0.5, 1.5, 0.5, -0.5])
    
    result = minmod(a, b)
    expected = np.array([0.5, 1.5, 0.0, 0.0])
    
    print("\n测试 Minmod 限制器 (数组):")
    print("输入 a:", a)
    print("输入 b:", b)
    print("结果:  ", result)
    print("预期:  ", expected)
    
    if np.allclose(result, expected):
        print("数组测试通过!")
    else:
        print("数组测试失败!")

def test_muscl_reconstruction():
    """测试 MUSCL 重构"""
    from schemes.tvd import muscl_reconstruction
    
    # 创建测试数据: 线性斜坡
    nx = 5
    x = np.linspace(0, 1, nx)
    U = np.array([x, x, x])  # 三个守恒变量
    
    print("\n测试 MUSCL 重构:")
    print("输入数据:", U[0])
    
    U_L, U_R = muscl_reconstruction(U, limiter='minmod')
    
    print("重构左状态:", U_L[0])
    print("重构右状态:", U_R[0])
    
    # 检查边界值
    if U_L[0,0] == U[0,0] and U_R[0,-1] == U[0,-1]:
        print("边界处理正确!")
    else:
        print("边界处理错误!")
    
    # 检查内部点重构
    print("内部点重构:")
    for i in range(1, nx-1):
        print(f"点 {i}: 原始={U[0,i]:.2f}, 左重构={U_L[0,i]:.2f}, 右重构={U_R[0,i]:.2f}")

def test_rk3():
    """测试三阶 Runge-Kutta"""
    from time_integration import rk3
    
    # 测试函数: dy/dt = -y
    def rhs(y):
        return -y
    
    # 边界条件函数 (空函数)
    def dummy_bc(y, params):
        return y
    
    # 初始条件
    y0 = np.array([1.0])
    dt = 0.1
    t_end = 1.0
    
    # 精确解
    exact = np.exp(-t_end)
    
    # 时间推进
    y = y0.copy()
    t = 0.0
    while t < t_end:
        dt_step = min(dt, t_end - t)
        y = rk3(y, rhs, dt_step, {'boundary_func': dummy_bc})
        t += dt_step
    
    error = abs(y[0] - exact)
    
    print("\n测试三阶 Runge-Kutta:")
    print(f"数值解: {y[0]:.6f}, 精确解: {exact:.6f}, 误差: {error:.2e}")
    
    if error < 1e-5:
        print("测试通过!")
    else:
        print("测试失败!")

if __name__ == "__main__":
    # 创建测试结果目录
    os.makedirs("test_results", exist_ok=True)
    
    # 运行单元测试
    test_minmod_limiter()
    test_muscl_reconstruction()
    test_rk3()
    
    # 运行主测试
    main()
    
    print("\n所有测试完成!")