import numpy as np
from initialization import create_domain, initialize_sod, set_initial_conditions
from utils.visualization import plot_solution_comparison
from utils.exact_solution import compute_exact

def main():
    # 设置计算域
    domain = create_domain(nx=400, x_min=-5.0, x_max=5.0, t_end=2.0)
    
    # 设置物理参数和初始条件
    physics = initialize_sod(domain)
    
    # 整合参数
    params = set_initial_conditions(physics, domain)
    
    # 输出关键参数
    print("计算域配置:")
    print(f"  空间范围: [{params['x_min']:.1f}, {params['x_max']:.1f}]")
    print(f"  网格点数: {params['nx']}, 空间步长: Δx = {params['dx']:.4f}")
    print(f"  时间范围: [0, {params['t_end']:.1f}], CFL数: {params['cfl']}")
    
    print("\n物理参数:")
    print(f"  比热比: γ = {params['gamma']}, 定容比热: Cv = {params['cv']}")
    
    print("\n初始条件:")
    print(f"  密度范围: min={np.min(params['rho_init']):.4f}, max={np.max(params['rho_init']):.4f}")
    print(f"  速度范围: min={np.min(params['u_init']):.4f}, max={np.max(params['u_init']):.4f}")
    print(f"  压强范围: min={np.min(params['p_init']):.4f}, max={np.max(params['p_init']):.4f}")
    
    if 'discontinuity_position' in params:
        disc_x = params['discontinuity_position']
        left = params['discontinuity_state']['left']
        right = params['discontinuity_state']['right']
        print(f"\n初始间断位置: x ≈ {disc_x:.2f}")
        print(f"  左侧状态: ρ={left['rho']:.3f}, u={left['u']:.3f}, p={left['p']:.3f}")
        print(f"  右侧状态: ρ={right['rho']:.3f}, u={right['u']:.3f}, p={right['p']:.3f}")

if __name__ == "__main__":
    main()