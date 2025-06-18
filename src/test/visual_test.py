import numpy as np
import matplotlib.pyplot as plt
from initialization.domain_setup import create_domain
from initialization.sod_initial import initialize_sod, set_initial_conditions
from utils.visualization import plot_solution_comparison, plot_error_analysis, plot_method_comparison

def generate_exact_solution(params, t):
    """
    生成精确解数据（模拟函数）
    
    参数:
    params -- 计算参数字典
    t -- 当前时间
    
    返回:
    精确解数据字典 {'x': x_exact, 'rho': rho_exact, 'u': u_exact, 'p': p_exact}
    """
    # 获取计算域参数
    x_min = params['x_min']
    x_max = params['x_max']
    nx = params['nx']
    
    # 创建精确解网格（通常比数值解更精细）
    x_exact = np.linspace(x_min, x_max, nx * 2)
    n_exact = len(x_exact)
    
    # 生成精确解物理量（这里使用解析表达式模拟精确解）
    gamma = params['gamma']
    
    # 密度精确解 - 模拟Sod问题的典型分布
    # 修复数组形状问题
    rho_exact = np.zeros(n_exact)
    for i in range(n_exact):
        if x_exact[i] < 0.5 * t:
            rho_exact[i] = 0.8
        elif x_exact[i] < 1.0 * t:
            # 在膨胀波区域线性过渡
            frac = (x_exact[i] - 0.5 * t) / (0.5 * t)
            rho_exact[i] = 0.8 - 0.6 * frac
        else:
            rho_exact[i] = 0.2
    
    # 速度精确解
    u_exact = np.zeros(n_exact)
    for i in range(n_exact):
        if x_exact[i] < 0.5 * t:
            u_exact[i] = 0.0
        elif x_exact[i] < 1.0 * t:
            # 在膨胀波区域线性过渡
            frac = (x_exact[i] - 0.5 * t) / (0.5 * t)
            u_exact[i] = 0.5 * frac
        else:
            u_exact[i] = 0.5
    
    # 压强精确解
    p_exact = np.zeros(n_exact)
    for i in range(n_exact):
        if x_exact[i] < 0.5 * t:
            p_exact[i] = 0.8
        elif x_exact[i] < 1.0 * t:
            # 在膨胀波区域线性过渡
            frac = (x_exact[i] - 0.5 * t) / (0.5 * t)
            p_exact[i] = 0.8 - 0.7 * frac
        else:
            p_exact[i] = 0.1
    
    return {
        'x': x_exact,
        'rho': rho_exact,
        'u': u_exact,
        'p': p_exact
    }

def generate_numerical_solution(params, exact_data, noise_level=0.05):
    """
    生成模拟数值解数据（在精确解基础上添加噪声）
    
    参数:
    params -- 计算参数字典
    exact_data -- 精确解数据字典
    noise_level -- 噪声水平 (默认0.05)
    
    返回:
    数值解守恒变量 U = [ρ, ρu, E]
    """
    # 获取数值解网格
    x_num = params['x']
    gamma = params['gamma']
    
    # 插值精确解到数值网格
    rho_exact = np.interp(x_num, exact_data['x'], exact_data['rho'])
    u_exact = np.interp(x_num, exact_data['x'], exact_data['u'])
    p_exact = np.interp(x_num, exact_data['x'], exact_data['p'])
    
    # 添加噪声模拟数值误差
    np.random.seed(42)  # 固定随机种子确保可重复性
    rho_num = rho_exact * (1 + noise_level * np.random.randn(len(x_num)))
    u_num = u_exact * (1 + noise_level * np.random.randn(len(x_num)))
    p_num = p_exact * (1 + noise_level * np.random.randn(len(x_num)))
    
    # 确保物理量有意义
    rho_num = np.clip(rho_num, 0.1, 1.0)
    u_num = np.clip(u_num, -0.1, 1.0)
    p_num = np.clip(p_num, 0.05, 1.0)
    
    # 计算守恒变量 [ρ, ρu, E]
    E = p_num / (gamma - 1) + 0.5 * rho_num * u_num**2
    U = np.array([rho_num, rho_num * u_num, E])
    
    return U

def test_single_comparison():
    """测试单个时间步的数值解与精确解对比"""
    print("="*70)
    print("测试1: 单时间步数值解与精确解对比")
    print("="*70)
    
    # 创建计算域和初始条件
    domain = create_domain(nx=200, x_min=-5.0, x_max=5.0, t_end=0.2)
    physics = initialize_sod(domain)
    params = set_initial_conditions(physics, domain)
    
    # 设置当前时间
    t = 0.2
    
    # 生成精确解
    exact_data = generate_exact_solution(params, t)
    
    # 生成数值解（添加噪声）
    U_num = generate_numerical_solution(params, exact_data, noise_level=0.03)
    
    # 绘制对比图
    plot_title = f"测试可视化模块 (t={t:.2f})"
    filename = "test_results/single_comparison.png"
    plot_solution_comparison(params, U_num, exact_data, t, title=plot_title, filename=filename)
    
    print("测试完成! 对比图已保存至 test_results/single_comparison.png")

def test_error_analysis():
    """测试误差收敛性分析"""
    print("\n" + "="*70)
    print("测试2: 误差收敛性分析")
    print("="*70)
    
    # 测试不同网格分辨率
    nx_values = [50, 100, 200, 400]
    errors = {'rho': [], 'u': [], 'p': []}
    t = 0.2
    
    for nx in nx_values:
        # 创建计算域
        domain = create_domain(nx=nx, x_min=-5.0, x_max=5.0, t_end=t)
        physics = initialize_sod(domain)
        params = set_initial_conditions(physics, domain)
        
        # 生成精确解
        exact_data = generate_exact_solution(params, t)
        
        # 生成数值解（噪声水平与网格无关）
        U_num = generate_numerical_solution(params, exact_data, noise_level=0.02)
        
        # 计算数值解原始变量
        gamma = params['gamma']
        rho_num = U_num[0]
        u_num = U_num[1] / rho_num
        p_num = (gamma - 1) * (U_num[2] - 0.5 * rho_num * u_num**2)
        
        # 插值精确解到数值网格
        rho_exact = np.interp(params['x'], exact_data['x'], exact_data['rho'])
        u_exact = np.interp(params['x'], exact_data['x'], exact_data['u'])
        p_exact = np.interp(params['x'], exact_data['x'], exact_data['p'])
        
        # 计算L2误差
        errors['rho'].append(np.sqrt(np.mean((rho_num - rho_exact)**2)))
        errors['u'].append(np.sqrt(np.mean((u_num - u_exact)**2)))
        errors['p'].append(np.sqrt(np.mean((p_num - p_exact)**2)))
        
        print(f"网格 nx={nx}: 密度误差={errors['rho'][-1]:.4f}, 速度误差={errors['u'][-1]:.4f}, 压强误差={errors['p'][-1]:.4f}")
    
    # 绘制误差分析图
    plot_error_analysis(params, errors, nx_values, "测试方法")
    print("测试完成! 误差分析图已保存至 test_results/测试方法_误差收敛性.png")

def test_method_comparison():
    """测试不同数值方法对比"""
    print("\n" + "="*70)
    print("测试3: 不同数值方法对比")
    print("="*70)
    
    # 创建计算域
    domain = create_domain(nx=200, x_min=-5.0, x_max=5.0, t_end=0.2)
    physics = initialize_sod(domain)
    params = set_initial_conditions(physics, domain)
    
    # 设置当前时间
    t = 0.2
    
    # 生成精确解
    exact_data = generate_exact_solution(params, t)
    
    # 创建模拟的不同方法结果
    comparison_data = {}
    methods = ["TVD+FVS", "WENO+Roe", "GVC+Character"]
    
    for method in methods:
        # 对于每种方法，生成略有不同的数值解
        noise_level = 0.02 if method == "TVD+FVS" else 0.01 if method == "WENO+Roe" else 0.015
        U_num = generate_numerical_solution(params, exact_data, noise_level)
        
        # 计算原始变量
        gamma = params['gamma']
        rho_num = U_num[0]
        u_num = U_num[1] / rho_num
        p_num = (gamma - 1) * (U_num[2] - 0.5 * rho_num * u_num**2)
        
        # 添加到对比数据
        comparison_data[method] = {
            'x': params['x'],
            'rho': rho_num,
            'u': u_num,
            'p': p_num
        }
        print(f"已生成 {method} 方法模拟数据")
    
    # 绘制方法对比图
    plot_method_comparison(comparison_data, t)
    print("测试完成! 方法对比图已保存至 test_results/数值方法对比_t=0.20.png")

if __name__ == "__main__":
    # 创建测试结果目录
    import os
    os.makedirs("test_results", exist_ok=True)
    
    # 运行测试
    test_single_comparison()
    test_error_analysis()
    test_method_comparison()
    
    print("\n所有测试完成! 请查看 test_results/ 目录下的输出图像")