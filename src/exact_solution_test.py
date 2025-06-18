import numpy as np
import matplotlib.pyplot as plt
from initialization.domain_setup import create_domain
from initialization.sod_initial import initialize_sod, set_initial_conditions
from utils.exact_solution import compute_exact_solution, interpolate_exact_to_grid
from utils.visualization import plot_solution_comparison, set_chinese_font
import utils.gitlab_sod_analytical as gsa  # 添加这行导入

def test_exact_solution():
    """测试精确解计算模块"""
    print("="*70)
    print("精确解模块测试")
    print("="*70)
    
    # 设置中文字体
    set_chinese_font()
    
    # 创建计算域和初始条件
    domain = create_domain(nx=2000, x_min=-5.0, x_max=5.0, t_end=1.0)
    physics = initialize_sod(domain)
    params = set_initial_conditions(physics, domain)
    
    # 设置不同时间点
    times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    if len(times) > 1:
        fig, axs = plt.subplots(len(times), 1, figsize=(10, 15), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        axs = [ax]
    
    for i, t in enumerate(times):
        # 计算精确解
        exact_data = compute_exact_solution(params, t)
        
        # 打印精确解信息
        print(f"\n时间 t={t:.2f} 精确解统计:")
        print(f"  密度: min={np.min(exact_data['rho']):.4f}, max={np.max(exact_data['rho']):.4f}")
        print(f"  速度: min={np.min(exact_data['u']):.4f}, max={np.max(exact_data['u']):.4f}")
        print(f"  压强: min={np.min(exact_data['p']):.4f}, max={np.max(exact_data['p']):.4f}")
        
        # 绘制精确解
        ax = axs[i]
        ax.plot(exact_data['x'], exact_data['rho'], 'b-', label='密度')
        ax.plot(exact_data['x'], exact_data['u'], 'r-', label='速度')
        ax.plot(exact_data['x'], exact_data['p'], 'g-', label='压强')
        ax.set_title(f'精确解 (t={t:.2f})')
        ax.set_ylabel('物理量值')
        ax.grid(True)
        ax.legend()
        
        # 标记关键位置
        positions, regions, _ = gsa.solve(  # 使用导入的 gsa 模块
            left_state=(1.0, 1.0, 0.0),
            right_state=(0.1, 0.125, 0.0),
            geometry=(-5.0, 5.0, 0.0),
            t=t,
            gamma=1.4
        )
        
        # 添加关键位置标记
        labels = ['膨胀波头', '膨胀波尾', '接触间断', '激波']
        colors = ['purple', 'cyan', 'orange', 'magenta']
        
        for j, pos in enumerate(positions):
            ax.axvline(x=pos, color=colors[j], linestyle='--', alpha=0.7)
            ax.text(pos, ax.get_ylim()[1]*0.9, labels[j], 
                   color=colors[j], fontsize=9, ha='center')
    
    plt.xlabel('x')
    plt.tight_layout()
    plt.savefig('test_results/exact_solution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n测试完成! 精确解分布图已保存至 test_results/exact_solution.png")

def test_exact_vs_numerical():
    """测试精确解与数值解对比"""
    print("\n" + "="*70)
    print("精确解与数值解对比测试")
    print("="*70)
    
    # 创建计算域和初始条件
    domain = create_domain(nx=200, x_min=-5.0, x_max=5.0, t_end=0.2)
    physics = initialize_sod(domain)
    params = set_initial_conditions(physics, domain)
    
    # 设置时间
    t = 0.2
    
    # 计算精确解
    exact_data = compute_exact_solution(params, t)
    
    # 生成"数值解" (这里使用精确解添加噪声来模拟数值解)
    np.random.seed(42)
    rho_num = exact_data['rho'] * (1 + 0.03 * np.random.randn(len(exact_data['x'])))
    u_num = exact_data['u'] * (1 + 0.03 * np.random.randn(len(exact_data['x'])))
    p_num = exact_data['p'] * (1 + 0.03 * np.random.randn(len(exact_data['x'])))
    
    # 计算守恒变量
    gamma = params['gamma']
    E_num = p_num / (gamma - 1) + 0.5 * rho_num * u_num**2
    U_num = np.array([rho_num, rho_num * u_num, E_num])
    
    # 创建模拟的数值解参数
    num_params = params.copy()
    num_params['x'] = exact_data['x']
    
    # 绘制对比图
    plot_title = f"精确解与数值解对比 (t={t:.2f})"
    filename = "test_results/exact_vs_numerical.png"
    plot_solution_comparison(num_params, U_num, exact_data, t, title=plot_title, filename=filename)
    
    print("测试完成! 对比图已保存至 test_results/exact_vs_numerical.png")

if __name__ == "__main__":
    # 创建测试结果目录
    import os
    os.makedirs("test_results", exist_ok=True)
    
    # 运行测试
    test_exact_solution()
    test_exact_vs_numerical()
    
    print("\n精确解模块测试完成!")