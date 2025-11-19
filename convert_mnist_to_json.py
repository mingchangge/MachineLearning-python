# convert_mnist_to_json.py
import tensorflow as tf
import json
import numpy as np

# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# # 归一化并转为 list（JSON 可序列化）
# x_train = (x_train.astype('float32') / 255.0).tolist()
# y_train = y_train.tolist()
# x_test = (x_test.astype('float32') / 255.0).tolist()
# y_test = y_test.tolist()

# with open('mnist.json', 'w') as f:
#     json.dump({
#         'x_train': x_train[:1000],   # 可限制数量加速
#         'y_train': y_train[:1000],
#         'x_test': x_test[:200],
#         'y_test': y_test[:200]
#     }, f)
# print("✅ mnist.json 已生成")


print("正在加载 MNIST 数据集...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 归一化到 [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 可选：限制数据量以加快训练（全量约 60k，这里用 6000 + 1000）
x_train = x_train[:6000]
y_train = y_train[:6000]
x_test = x_test[:1000]
y_test = y_test[:1000]

# 转为 Python list（JSON 可序列化）
data = {
    "x_train": x_train.tolist(),
    "y_train": y_train.tolist(),
    "x_test": x_test.tolist(),
    "y_test": y_test.tolist()
}

# 保存为 JSON
with open("mnist.json", "w") as f:
    json.dump(data, f)

print(f"✅ 已保存 {len(x_train)} 条训练数据 和 {len(x_test)} 条测试数据 到 mnist.json")