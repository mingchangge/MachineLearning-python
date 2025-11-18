import matplotlib.pyplot as plt
import numpy as np

# === 新增：解决中文乱码 ===
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']  # 支持中文的字体列表
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号（如 -1）
# =========================

# 1. 定义一个简单的“笑脸”：用5个点表示（眼睛、嘴）
points = np.array([
    [-1, 1],   # 左眼
    [1, 1],    # 右眼
    [-1, -1],  # 嘴左
    [0, -1.5], # 嘴中
    [1, -1]    # 嘴右
]).T  # 转置成 2×5 矩阵（每列是一个点）,原本每行一个点，现在变成：第一行是所有 x 坐标，第二行是所有 y 坐标 → 方便矩阵运算。

# 2. 定义拉伸矩阵：纵向拉伸2倍
A = np.array([[1, 0],
              [0, 2]])

# 3. 应用矩阵：新点 = A × 原点
new_points = A @ points  # @ 是矩阵乘法

# 2.1 横向拉伸2倍
A1=np.array([[2,0],[0,1]])
# 3.1 应用矩阵：新点 = A1 × 原点
new_points1=A1@points

# 2.2 逆时针旋转90度
A2=np.array([[0,-1],[1,0]])
# 3.2 应用矩阵：新点 = A2 × 原点
new_points2=A2@points

# 2.3 斜着切
A3=np.array([[1,0.5],[0,1]])
# 3.3 应用矩阵：新点 = A3 × 原点
new_points3=A3@points



# 4. 画图对比
plt.figure(figsize=(16, 8))

# 原始笑脸
plt.subplot(1, 5, 1) #准备画5个图，这是第一个
plt.scatter(points[0], points[1], c='blue', s=100) # 在 (x,y) 位置画一个点（scatter = 散点），points[0]	取第一行 → 所有点的 x 坐标，points[1]	取第二行 → 所有点的 y 坐标
plt.title("原始笑脸")
plt.axis('equal') #让 x 和 y 的单位长度一样（否则圆会变椭圆）
plt.grid(True) # 显示网格线

# 纵向拉伸后的笑脸
plt.subplot(1, 5, 2)
plt.scatter(new_points[0], new_points[1], c='red', s=100)
plt.title("纵向拉伸2倍后")
plt.axis('equal')
plt.grid(True)
## 横向拉伸2倍后的笑脸
plt.subplot(1, 5, 3)
plt.scatter(new_points1[0], new_points1[1], c='green', s=100)
plt.title("横向拉伸2倍后")
plt.axis('equal')
plt.grid(True)

## 逆时针旋转90度后的笑脸
plt.subplot(1, 5, 4)
plt.scatter(new_points2[0], new_points2[1], c='orange', s=100)
plt.title("逆时针旋转90度后")
plt.axis('equal')
plt.grid(True)
## 斜着切后的笑脸
plt.subplot(1, 5, 5)
plt.scatter(new_points3[0], new_points3[1], c='purple', s=100)
plt.title("斜着切后")
plt.axis('equal')
plt.grid(True)

plt.show()