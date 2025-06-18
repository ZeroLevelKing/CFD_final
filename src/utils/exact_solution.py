from .gitlab_sod_analytical import solve as gitlab_solve

def exact_solution(params, t):
    """调用GitLab项目的精确解计算"""
    pos, regions, vals = gitlab_solve(
        left_state=(1.0, 1.0, 0.0), 
        right_state=(0.125, 0.125, 0.0),
        geometry=(params['x_min'], params['x_max'], 0),
        t=t,
        gamma=params['gamma']
    )
    return vals['x'], vals['rho'], vals['u'], vals['p']