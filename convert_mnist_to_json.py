# convert_mnist_to_json.py
import tensorflow as tf
import json
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 归一化并转为 list（JSON 可序列化）
x_train = (x_train.astype('float32') / 255.0).tolist()
y_train = y_train.tolist()
x_test = (x_test.astype('float32') / 255.0).tolist()
y_test = y_test.tolist()

with open('mnist.json', 'w') as f:
    json.dump({
        'x_train': x_train[:1000],   # 可限制数量加速
        'y_train': y_train[:1000],
        'x_test': x_test[:200],
        'y_test': y_test[:200]
    }, f)
print("✅ mnist.json 已生成")