import numpy as np
import matplotlib.pyplot as plt
from initialization.domain_setup import create_domain
from initialization.sod_initial import initialize_sod, set_initial_conditions
from utils.boundary import apply_boundary_conditions, add_ghost_cells, remove_ghost_cells
from utils.visualization import set_chinese_font  # 添加中文支持

# 设置中文字体
set_chinese_font()

def test_boundary_conditions():
    """测试各种边界条件"""
    print("="*70)
    print("边界条件模块测试")
    print("="*70)
    
    # 创建计算域和初始条件
    domain = create_domain(nx=10, x_min=-5.0, x_max=5.0, t_end=0.2)
    physics = initialize_sod(domain)
    params = set_initial_conditions(physics, domain)
    
    # 获取初始守恒变量
    U = params['U_init']
    
    # 添加虚单元
    num_ghost = 2
    U_extended = add_ghost_cells(U, num_ghost)
    
    # 测试不同边界条件
    bc_types = ['non-reflective', 'periodic', 'fixed']
    results = {}
    
    for bc_type in bc_types:
        print(f"\n测试 {bc_type} 边界条件:")
        
        # 应用边界条件
        U_bc = apply_boundary_conditions(U_extended.copy(), params, num_ghost)
        
        # 移除虚单元以便比较
        U_internal = remove_ghost_cells(U_bc, num_ghost)
        
        # 保存结果
        results[bc_type] = U_bc
        
        # 打印边界值
        print("左边界虚单元值:")
        for i in range(num_ghost):
            print(f"  虚单元 {i}: ρ={U_bc[0,i]:.3f}, u={U_bc[1,i]/U_bc[0,i]:.3f}, p={(params['gamma']-1)*(U_bc[2,i]-0.5*U_bc[0,i]*(U_bc[1,i]/U_bc[0,i])**2):.3f}")
        
        print("右边界虚单元值:")
        n = U_bc.shape[1]
        for i in range(num_ghost):
            idx = n - 1 - i
            print(f"  虚单元 {idx}: ρ={U_bc[0,idx]:.3f}, u={U_bc[1,idx]/U_bc[0,idx]:.3f}, p={(params['gamma']-1)*(U_bc[2,idx]-0.5*U_bc[0,idx]*(U_bc[1,idx]/U_bc[0,idx])**2):.3f}")
    
    # 可视化结果
    plot_boundary_comparison(params, results, num_ghost)
    
    print("\n测试完成! 边界条件对比图已保存至 test_results/boundary_comparison.png")

def plot_boundary_comparison(params, results, num_ghost):
    """
    绘制不同边界条件下的虚单元值对比
    
    参数:
    params -- 计算参数字典
    results -- 不同边界条件下的结果字典 {bc_type: U_bc}
    num_ghost -- 虚单元层数
    """
    plt.figure(figsize=(15, 10))
    
    # 获取计算域坐标
    x_internal = params['x']
    x_min = params['x_min']
    x_max = params['x_max']
    dx = params['dx']
    
    # 创建包含虚单元的坐标
    x_left_ghost = np.linspace(x_min - num_ghost * dx, x_min - dx, num_ghost)
    x_right_ghost = np.linspace(x_max + dx, x_max + num_ghost * dx, num_ghost)
    x_full = np.concatenate([x_left_ghost, x_internal, x_right_ghost])
    
    # 创建子图
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 1, 3)
    
    # 绘制不同边界条件的结果
    colors = ['b', 'g', 'r', 'c', 'm']
    for i, (bc_type, U_bc) in enumerate(results.items()):
        # 计算原始变量
        rho = U_bc[0]
        u = U_bc[1] / rho
        p = (params['gamma'] - 1) * (U_bc[2] - 0.5 * rho * u**2)
        
        # 密度
        ax1.plot(x_full, rho, colors[i]+'-', linewidth=1.5, label=bc_type)
        
        # 速度
        ax2.plot(x_full, u, colors[i]+'-', linewidth=1.5, label=bc_type)
        
        # 压强
        ax3.plot(x_full, p, colors[i]+'-', linewidth=1.5, label=bc_type)
    
    # 标记虚单元区域
    ax1.axvspan(x_min - num_ghost * dx, x_min, alpha=0.2, color='gray', label='左虚单元')
    ax1.axvspan(x_max, x_max + num_ghost * dx, alpha=0.2, color='gray', label='右虚单元')
    ax2.axvspan(x_min - num_ghost * dx, x_min, alpha=0.2, color='gray')
    ax2.axvspan(x_max, x_max + num_ghost * dx, alpha=0.2, color='gray')
    ax3.axvspan(x_min - num_ghost * dx, x_min, alpha=0.2, color='gray')
    ax3.axvspan(x_max, x_max + num_ghost * dx, alpha=0.2, color='gray')
    
    # 设置标题和标签
    ax1.set_title('不同边界条件下的密度分布')
    ax1.set_ylabel(r'$\rho$')
    ax1.legend()
    ax1.grid(True)
    
    ax2.set_title('不同边界条件下的速度分布')
    ax2.set_ylabel(r'$u$')
    ax2.legend()
    ax2.grid(True)
    
    ax3.set_title('不同边界条件下的压强分布')
    ax3.set_ylabel(r'$p$')
    ax3.set_xlabel(r'$x$')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('test_results/boundary_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # 创建测试结果目录
    import os
    os.makedirs("test_results", exist_ok=True)
    
    # 运行测试
    test_boundary_conditions()
    print("\n边界条件测试完成!")