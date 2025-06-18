import numpy as np
from . import gitlab_sod_analytical as gsa

def compute_exact_solution(params, t):
    """
    计算Sod激波管问题的精确解
    
    参数:
    params -- 计算参数字典
    t -- 当前时间
    
    返回:
    精确解数据字典 {'x': x_exact, 'rho': rho_exact, 'u': u_exact, 'p': p_exact}
    """
    # 从参数中提取初始状态
    gamma = params['gamma']
    x_min = params['x_min']
    x_max = params['x_max']
    
    # Sod问题的标准初始状态
    left_state = (1.0, 1.0, 0.0)   # (pl, rhol, ul)
    right_state = (0.1, 0.125, 0.0) # (pr, rhor, ur)
    
    # 初始间断位置 (在0处)
    xi = 0.0
    
    # 计算精确解
    positions, regions, val_dict = gsa.solve(
        left_state=left_state,
        right_state=right_state,
        geometry=(x_min, x_max, xi),
        t=t,
        gamma=gamma,
        npts=params['nx'] * 2  # 使用更精细的网格
    )
    
    # 返回精确解数据
    return {
        'x': val_dict['x'],
        'rho': val_dict['rho'],
        'u': val_dict['u'],
        'p': val_dict['p']
    }

def interpolate_exact_to_grid(params, exact_data):
    """
    将精确解数据插值到数值网格上
    
    参数:
    params -- 计算参数字典
    exact_data -- 精确解数据字典
    
    返回:
    插值后的精确解数据 {'x': params['x'], 'rho': rho_interp, 'u': u_interp, 'p': p_interp}
    """
    from scipy.interpolate import interp1d
    
    # 数值网格坐标
    x_num = params['x']
    
    # 创建插值函数
    rho_interp = interp1d(exact_data['x'], exact_data['rho'], kind='linear')
    u_interp = interp1d(exact_data['x'], exact_data['u'], kind='linear')
    p_interp = interp1d(exact_data['x'], exact_data['p'], kind='linear')
    
    # 在数值网格上插值
    return {
        'x': x_num,
        'rho': rho_interp(x_num),
        'u': u_interp(x_num),
        'p': p_interp(x_num)
    }

def calculate_error(U_num, exact_data, params):
    """
    计算数值解与精确解之间的误差
    
    参数:
    U_num -- 数值解的守恒变量数组 [ρ, ρu, E]
    exact_data -- 精确解数据字典 (已插值到数值网格)
    params -- 计算参数字典
    
    返回:
    误差字典 {'rho': L2_error_rho, 'u': L2_error_u, 'p': L2_error_p}
    """
    # 从守恒变量计算数值解的原始变量
    rho_num = U_num[0]
    u_num = U_num[1] / rho_num
    p_num = (params['gamma'] - 1) * (U_num[2] - 0.5 * rho_num * u_num**2)
    
    # 提取精确解值
    rho_exact = exact_data['rho']
    u_exact = exact_data['u']
    p_exact = exact_data['p']
    
    # 计算L2误差
    error_rho = np.sqrt(np.mean((rho_num - rho_exact)**2))
    error_u = np.sqrt(np.mean((u_num - u_exact)**2))
    error_p = np.sqrt(np.mean((p_num - p_exact)**2))
    
    return {
        'rho': error_rho,
        'u': error_u,
        'p': error_p
    }

def calculate_convergence_rate(errors, nxs):
    """
    计算收敛率
    
    参数:
    errors -- 不同网格分辨率下的误差字典 {'rho': [error1, error2, ...], ...}
    nxs -- 网格分辨率数组 [nx1, nx2, ...]
    
    返回:
    收敛率字典 {'rho': rate_rho, 'u': rate_u, 'p': rate_p}
    """
    rates = {}
    
    for key in errors.keys():
        rates[key] = []
        for i in range(1, len(errors[key])):
            ratio = nxs[i-1] / nxs[i]
            error_ratio = errors[key][i-1] / errors[key][i]
            rate = np.log(error_ratio) / np.log(ratio)
            rates[key].append(rate)
    
    return rates

def get_region_names(regions_data):
    """
    获取区域名称描述
    
    参数:
    regions_data -- 精确解区域信息
    
    返回:
    区域名称列表
    """
    return list(regions_data.keys())