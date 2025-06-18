import numpy as np

def initialize_sod(domain, gamma=1.4, cv=1.0):
    """
    设置Sod激波管问题的初始条件
    
    参数:
    domain -- 计算域参数字典
    gamma -- 比热比 (默认1.4, 空气)
    cv -- 定容比热容 (默认1.0, 无量纲化)
    
    返回:
    包含所有物理参数的字典
    """
    x = domain['x']
    
    # Sod激波管初始条件
    rho = np.where(x < 0, 1.0, 0.125)  # 密度
    u = np.zeros_like(x)               # 速度
    p = np.where(x < 0, 1.0, 0.1)      # 压强
    
    # 计算守恒变量 [ρ, ρu, E]
    E = p / (gamma - 1) + 0.5 * rho * u**2
    U = np.array([rho, rho * u, E])
    
    # 收集所有物理参数
    physics = {
        'gamma': gamma,
        'cv': cv,
        'U_init': U,
        'rho_init': rho,
        'u_init': u,
        'p_init': p,
        'bc_type': 'non-reflective',  # 无反射边界
        'cfl': 0.5,  # 默认CFL数
    }
    
    return physics

def set_initial_conditions(physics, domain):
    """
    设置初始条件并整合到计算域参数中
    
    参数:
    physics -- 物理参数字典
    domain -- 计算域参数字典
    
    返回:
    包含所有计算参数的完整字典
    """
    # 整合物理参数和计算域参数
    params = {**domain, **physics}
    
    # 添加初始诊断信息
    discontinuity_index = np.argmax(np.abs(np.diff(params['rho_init'])) > 0.5)
    if discontinuity_index < len(params['x'])-1:
        params['discontinuity_position'] = params['x'][discontinuity_index]
        params['discontinuity_state'] = {
            'left': {
                'rho': params['rho_init'][discontinuity_index],
                'u': params['u_init'][discontinuity_index],
                'p': params['p_init'][discontinuity_index]
            },
            'right': {
                'rho': params['rho_init'][discontinuity_index+1],
                'u': params['u_init'][discontinuity_index+1],
                'p': params['p_init'][discontinuity_index+1]
            }
        }
    
    return params