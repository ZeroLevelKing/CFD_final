# 导入边界条件模块
from .boundary import (
   apply_boundary_conditions,
    non_reflective_bc,
    characteristic_bc,
    periodic_bc,
    fixed_bc,
    add_ghost_cells,
    remove_ghost_cells
)

# 导入精确解计算模块
from .exact_solution import (
    compute_exact_solution,
    interpolate_exact_to_grid,
    calculate_error,
    calculate_convergence_rate,
    get_region_names
)

# 导入可视化模块
from .visualization import (
    plot_solution_comparison,
    plot_error_analysis,
    plot_method_comparison,
    set_chinese_font
)

