import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 设置时间
t = 1.0  # 任意时间单位

# 定义初始条件
rho_L = 1.0  # 左侧密度
u_L = 0.0    # 左侧速度
p_L = 1.0    # 左侧压力

rho_R = 0.125  # 右侧密度
u_R = 0.0      # 右侧速度
p_R = 0.1      # 右侧压力

gamma = 1.4  # 比热比

# 计算关键位置 (使用典型值)
c_L = np.sqrt(gamma * p_L / rho_L)  # 左侧声速
u_star = 0.927  # 接触间断速度 (典型值)
p_star = 0.303  # 中间压力 (典型值)
rho_2 = 0.426   # 区域2密度 (典型值)
rho_3 = 0.265   # 区域3密度 (典型值)

# 计算波位置
x_left = -c_L * t
x_contact = u_star * t
x_shock = (u_star + c_L) * t  # 简化计算激波位置

# 绘制区域分界线
ax.axvline(x=x_left, color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
ax.axvline(x=x_contact, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
ax.axvline(x=x_shock, color='red', linestyle='-', linewidth=1.5, alpha=0.7)

# 绘制特征线（膨胀波）
x_fan = np.linspace(x_left, x_contact, 50)
c_fan = np.linspace(c_L, c_L*0.8, 50)  # 简化声速变化
y_fan = 0.1 * (c_fan / c_L)  # 用声速变化表示膨胀波
ax.plot(x_fan, y_fan, 'purple', linestyle='--', alpha=0.7)



# 添加波标签
ax.annotate('膨胀波', xy=(x_left, 0.2), xytext=(x_left-0.5, 0.4), 
            arrowprops=dict(facecolor='purple', shrink=0.05, width=1, headwidth=8),
            fontsize=10, color='purple')
ax.annotate('接触间断', xy=(x_contact, 0.1), xytext=(x_contact+0.5, 0.3), 
            arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=8),
            fontsize=10, color='green')
ax.annotate('激波', xy=(x_shock, 0.1), xytext=(x_shock+0.5, 0.3), 
            arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
            fontsize=10, color='red')

# 添加时间标记
ax.text(1.5, -0.15, f'时间 t = {t}', fontsize=10)

# 设置坐标轴
ax.set_xlim(-3.0, 4.0)
ax.set_ylim(-0.2, 1.0)
ax.set_xlabel('位置 x')
ax.set_title('Sod激波管波系结构 (t > 0)')
ax.grid(True, linestyle='--', alpha=0.6)

# 移除Y轴刻度
ax.set_yticks([])

# 添加图例说明
ax.text(-2.5, 0.9, '← 未扰动高压区', fontsize=9, color='blue')
ax.text(2.5, 0.9, '未扰动低压区 →', fontsize=9, color='red', ha='right')

# 保存图像
plt.tight_layout()
plt.show()