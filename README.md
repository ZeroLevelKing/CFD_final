# CFD_final
The final project of CFD

src/
├── main.py                # 主程序入口
├── config.py              # 全局配置参数
├── time_integration.py    # 时间推进方法
│
├── initialization/        # 初始化模块
│   ├── __init__.py
│   ├── domain_setup.py    # 计算域设置
│   └── sod_initial.py     # Sod初始条件
│
├── fluxes/                # 通量计算方法
│   ├── __init__.py
│   ├── fvs.py             # 通量矢量分裂
│   └── fds.py             # 通量差分裂
│
├── schemes/               # 激波捕捉格式
│   ├── __init__.py
│   ├── tvd.py             # TVD格式实现
│   ├── gvc.py             # 群速度控制
│   └── weno.py            # WENO格式
│
├── utils/                 # 工具函数
│   ├── __init__.py
│   ├── boundary.py        # 边界条件处理
│   ├── exact_solution.py  # 精确解计算
│   ├── visualization.py   # 结果可视化
│   └── diagnostics.py     # 计算诊断
│
└── characteristic/        # 特征重构方法(附加题)
    ├── __init__.py
    ├── projection.py      # 特征投影
    └── reconstruction.py  # 特征空间重构