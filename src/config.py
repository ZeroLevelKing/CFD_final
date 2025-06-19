import numpy as np

class Config:
    def __init__(self):
        # 计算域参数
        self.nx = 200
        self.x_min = -5.0
        self.x_max = 5.0
        self.t_end = 1.0
        
        # 物理参数
        self.gamma = 1.4
        self.cv = 1.0
        
        # 数值方法选择
        self.scheme = 'tvd'       # 激波捕捉格式: 'tvd', 'gvc', 'weno'
        self.flux_method = 'fvs'   # 通量处理方法: 'fvs', 'fds'
        self.flux_type = 'van_leer'  
                                   # 具体通量类型: 
                                   # FVS: 'steger_warming', 'van_leer', 'ausm', 'lax_friedrichs'
                                   # FDS: 'hll', 'lax_wendroff', 'roe（存在bug）'
        self.limiter = 'minmod'    # TVD限制器类型
        self.weno_variant = 'z'    # WENO变体: 'js', 'z'
        
        # 边界条件
        self.bc_type = 'non-reflective'  # 'non-reflective', 'periodic', 'fixed'
        self.num_ghost = 3                # 虚单元层数
        
        # 时间步进
        self.cfl = 0.5
        
        # 输出控制
        self.output_interval = 0.1  # 输出间隔时间
        self.save_plots = True       # 是否保存图像
        self.plot_dir = "results"    # 图像保存目录
    
    def create_domain(self):
        """创建计算域"""
        return {
            'nx': self.nx,
            'dx': (self.x_max - self.x_min) / self.nx,
            'x': np.linspace(self.x_min, self.x_max, self.nx),
            'x_min': self.x_min,
            'x_max': self.x_max,
            't_end': self.t_end,
            'dt': 0.0  # 将由CFL条件计算
        }

params = Config()