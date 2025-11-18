# 极简代码演示：一个“两层小神经网络” + 本节重点：ReLU 激活函数

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 输入：嘴的3个点（直线）
X = np.array([
    [-1, -1],   # 嘴左
    [0, -1],    # 嘴中（平的）
    [1, -1]     # 嘴右
])  # shape: (3, 2)

# 目标：希望嘴变成向下弯曲的弧线（比如抛物线 y = -0.5*x² -1）
x_vals = X[:, 0]
y_target = 0.5 * x_vals**2 + 0.5  # 弯曲的嘴
Y_true = np.column_stack([x_vals, y_target])  # 保持x不变，只变y

# 准备数据
X_train = X          # 输入：3个点 (3,2)
y_train = Y_true     # 目标：弯曲嘴 (3,2)

# 创建两个模型：一个带 ReLU，一个无激活（线性）
model_relu = MLPRegressor(
    # hidden_layer_sizes=(10,),  # 1个隐藏层，10个神经元
    hidden_layer_sizes=(100,),  # 更多神经元，拟合更光滑
    activation='relu',
    # max_iter=1000,
    max_iter=2000,
    # random_state=0,
    random_state=42,
    solver='adam'
)

model_linear = MLPRegressor(
    # hidden_layer_sizes=(10,),
    hidden_layer_sizes=(20,),
    activation='identity',  # 线性激活（即无激活）
    # max_iter=1000,
    max_iter=2000,
    # random_state=0,
    random_state=42,
    solver='adam'
)

# 训练
model_relu.fit(X_train, y_train)
model_linear.fit(X_train, y_train)

# 预测
# y_pred_relu = model_relu.predict(X_train)
# y_pred_linear = model_linear.predict(X_train)

# 预测密集点，画完整曲线
x_dense = np.linspace(-1.2, 1.2, 100).reshape(-1, 1)
y_flat = np.full_like(x_dense, -1)  # 输入的 y 始终是 -1（原始是平嘴）
X_dense = np.hstack([x_dense, y_flat])  # shape: (100, 2)

y_pred_relu = model_relu.predict(X_dense)
y_pred_linear = model_linear.predict(X_dense)


plt.figure(figsize=(12, 4))

# 原始输入
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='black', s=100, label='输入（直线嘴）')
plt.title("原始输入")
plt.grid(True); plt.axis('equal')

# ReLU 模型
plt.subplot(1, 3, 2)
plt.scatter(y_pred_relu[:, 0], y_pred_relu[:, 1], c='red', s=100, label='ReLU 输出')
plt.scatter(Y_true[:, 0], Y_true[:, 1], c='green', s=50, marker='x', label='目标（弯曲嘴）')
plt.title("带 ReLU → 能拟合弯曲")
plt.legend(); plt.grid(True); plt.axis('equal')

# 线性模型
plt.subplot(1, 3, 3)
plt.scatter(y_pred_linear[:, 0], y_pred_linear[:, 1], c='blue', s=100, label='线性输出')
plt.scatter(Y_true[:, 0], Y_true[:, 1], c='green', s=50, marker='x', label='目标')
plt.title("无线性激活 → 仍是直线")
plt.legend(); plt.grid(True); plt.axis('equal')

plt.tight_layout()
plt.show()
