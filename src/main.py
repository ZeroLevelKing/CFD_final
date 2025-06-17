from func import initialize

# 使用默认参数初始化
params = initialize()

# 使用自定义参数初始化
custom_params = initialize(nx=400, t_end=1.5, cfl=0.3)