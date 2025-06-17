# main.py
import numpy as np
from func import initialize, plot_solution
import matplotlib.pyplot as plt
import os

def main():
    """
    主函数：初始化Sod激波管问题并绘制初始场分布
    """
    print("="*70)
    print("计算流体力学大作业 - Sod激波管问题求解")
    print("作者: 朱林")
    print("学号: 2200011028")
    print("="*70)
    
    # 初始化计算参数
    print("\n[步骤1] 初始化计算参数和网格...")
    params = initialize(nx=400)
    
    # 输出关键参数信息
    print(f"计算域: [{params['x_min']:.1f}, {params['x_max']:.1f}]")
    print(f"网格点数: {params['nx']}")
    print(f"空间步长: Δx = {params['dx']:.4f}")
    print(f"结束时间: t_end = {params['t_end']:.1f}")
    print(f"CFL数: {params['cfl']}")
    print(f"比热比: γ = {params['gamma']}")
    
    # 提取初始场
    U_init = params['U_init']
    x = params['x']
    
    # 计算初始场的物理量
    rho = U_init[0]
    u = U_init[1] / rho
    p = (params['gamma'] - 1) * (U_init[2] - 0.5 * rho * u**2)
    
    # 输出初始场统计信息
    print("\n[步骤2] 初始场统计信息:")
    print(f"密度: min={np.min(rho):.4f}, max={np.max(rho):.4f}, mean={np.mean(rho):.4f}")
    print(f"速度: min={np.min(u):.4f}, max={np.max(u):.4f}, mean={np.mean(u):.4f}")
    print(f"压强: min={np.min(p):.4f}, max={np.max(p):.4f}, mean={np.mean(p):.4f}")
    
    # 绘制初始场
    print("\n[步骤3] 绘制初始场分布图...")
    plot_solution(params, U_init, t=0.0, 
                 title='Sod激波管问题初始场分布',filename='sod_initial.png')
    
    # 添加额外诊断输出
    discontinuity_index = np.argmax(np.abs(np.diff(rho)) > 0.5)
    if discontinuity_index < len(x)-1:
        print(f"\n检测到初始间断位置: x ≈ {x[discontinuity_index]:.2f}")
        print(f"  左侧: ρ={rho[discontinuity_index]:.3f}, u={u[discontinuity_index]:.3f}, p={p[discontinuity_index]:.3f}")
        print(f"  右侧: ρ={rho[discontinuity_index+1]:.3f}, u={u[discontinuity_index+1]:.3f}, p={p[discontinuity_index+1]:.3f}")
    
    # 输出成功信息
    print("\n" + "="*70)
    print("初始场可视化完成!")
    print("="*70)
    

if __name__ == "__main__":
    main()