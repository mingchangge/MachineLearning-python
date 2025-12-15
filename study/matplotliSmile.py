# 使用matplotlib绘制笑脸

import numpy as np
import matplotlib.pyplot as plt

# 圆的参数方程生成圆上的点
def circle_points(cx, cy, r, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    return x, y

# 创建图像
fig, ax = plt.subplots(figsize=(6, 6))

# 绘制左眼
x_left_eye, y_left_eye = circle_points(-0.5, 0.5, 0.1)
ax.plot(x_left_eye, y_left_eye, 'k')

# 绘制右眼
x_right_eye, y_right_eye = circle_points(0.5, 0.5, 0.1)
ax.plot(x_right_eye, y_right_eye, 'k')

# 绘制微笑的嘴巴
x_mouth = np.linspace(-1, 1, 100)
y_mouth = 0.5 * x_mouth**2 - 0.5 # 向上弯曲的抛物线
ax.plot(x_mouth, y_mouth, 'k')

# 设置坐标轴等比例显示
ax.set_aspect('equal')
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])

# 移除坐标轴
plt.axis('off')

plt.show()