# 极简代码演示：一个“两层小神经网络”

import numpy as np
import matplotlib.pyplot as plt

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 1. 输入：5个点（每个点是2维）
X = np.array([
    [-1, 1],    # 左眼
    [1, 1],     # 右眼
    [-1, -1],   # 嘴左
    [0, -1.5],  # 嘴中
    [1, -1]     # 嘴右
])  # shape: (5, 2)

# 2. 第一层权重（随机初始化）：2维 → 3维
W1 = np.random.randn(2, 3) * 0.5
b1 = np.zeros(3)

# 3. 第二层权重：3维 → 2维（比如分类：眼睛 / 嘴巴）
W2 = np.random.randn(3, 2) * 0.5
b2 = np.zeros(2)

# 4. 前向传播（forward pass）
def relu(x):
    return np.maximum(0, x)  # 激活函数：负数变0，正数不变

# 第一层：X @ W1 + b1 → 激活
hidden = relu(X @ W1 + b1)   # shape: (5, 3)
# hidden = X @ W1 + b1 # 不使用 ReLU 会导致输出层的点云分布在一个平面上，而不是弯曲的笑脸轮廓。
# 第二层：hidden @ W2 + b2
output = hidden @ W2 + b2    # shape: (5, 2)

# 5. 画图对比
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c='blue', s=100)
plt.title("原始输入")
plt.axis('equal')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(hidden[:, 0], hidden[:, 1], c='green', s=100)
plt.title("第一层后（3维，只画前2维）")
plt.axis('equal')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(output[:, 0], output[:, 1], c='red', s=100)
plt.title("输出层")
plt.axis('equal')
plt.grid(True)

plt.tight_layout()
plt.show()



# 固定随机种子，让结果可重现
np.random.seed(42)

X2 = np.array([[-1, 1], [1, 1], [-1, -1], [0, -1.5], [1, -1]])
W2 = np.random.randn(2, 3) * 0.5
b2 = np.zeros(3)

print("X2 =\n", X2)
print("\nW2 =\n", W2)
print("\nX2 @ W2 =\n", X2 @ W2)
print("\nAfter ReLU =\n", np.maximum(0, X2 @ W2))
