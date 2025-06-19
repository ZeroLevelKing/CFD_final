# 计算流体力学期末大作业 - Sod激波管问题求解

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 项目概述
本项目实现了Sod激波管问题的数值求解，通过多种数值方法求解一维欧拉方程，并与精确解进行比较。项目包含完整的数值求解框架，支持多种激波捕捉格式、通量处理方法和时间积分方案。

## 项目结构
```
├──doc/                          # 存放报告和报告源码
│   ├── figure/                  # 存放报告中涉及的图片
│   ├── final.pdf                # 报告
│   ├── reference.bib            # 引用文献
│   └── final.tex                # 报告latex源码
├──src/
│   ├── main.py                  # 主程序入口
│   ├── characteristic.py        # 附加题单独程序
│   ├── config.py                # 参数配置模块
│   ├── time_integration.py      # 时间积分模块
│   ├── initialization/          # 初始条件模块
│   │   ├── domain_setup.py      # 计算域和网格设置
│   │   └── sod_initial.py       # Sod问题初始条件
│   ├── utils/                   # 工具函数库
│   │   ├── boundary.py          # 边界条件处理
│   │   ├── exact_solution.py    # 精确解计算
│   │   ├── gitlab_sod_analytical.py
│   │   └── visualization.py     # 结果可视化
│   ├── flux/                    # 通量计算模块
│   │   ├── flux_fvs.py          # FVS通量分裂方法
│   │   └── flux_fds.py          # FDS通量差分方法
│   ├── schemes/                 # 数值格式模块
│   │   ├── tvd.py               # TVD格式实现
│   │   ├── weno.py              # WENO格式实现
│   │   └── gvc.py               # 群速度控制格式
│   ├── test/                    # 单元测试目录
│   │   ├── initialization_test.py
│   │   ├── boundary_test.py
│   │   ├── tvd_fvs_rk3_test.py
│   │   ├── weno_test.py
│   │   ├── gvc_test.py
│   │   └── fds_test.py
│   └── result/                  # 结果输出目录
├──.gitignore
└──README.md

```

## 前置要求

- Python 3.7+
- matplotlib  第三方库
- numpy 第三方库
- dataclasses 第三方库
- texlive/simple Tex


## 安装与运行
查看报告：
在 ```doc```目录下，```final.pdf```即为报告
运行主程序：
将工作目录调整至 ``` src ```目录下，在shell中输入
   ```bash
   python main.py
   ```
查看附加题的尝试：
   ```bash
   python characteristic.py
   ```

## 参数配置
在`config.py`中可配置以下参数：
```python
# 计算域设置
DOMAIN = [-5, 5]        # 计算域范围 [x_min, x_max]
T_FINAL = 2.0           # 计算终止时间
N_POINTS = 200          # 网格点数
CFL = 0.8               # CFL数

# 初始条件
RHO_L, U_L, P_L = 1.0, 0.0, 1.0    # 左侧状态
RHO_R, U_R, P_R = 0.125, 0.0, 0.1  # 右侧状态
GAMMA = 1.4             # 比热比

# 数值方法选择
SCHEME = 'tvd'          # 可选: 'tvd', 'weno', 'gvc'
FLUX_METHOD = 'fvs_vanleer' # 可选: 'fvs_steger', 'fvs_vanleer', 'fvs_ausm', 'fds_hll'
...
```

## 输出结果
程序运行后将在`result/`目录下生成图像
