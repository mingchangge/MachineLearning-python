import tensorflow as tf

# 1. 加载 MNIST 数据集
print("正在加载 MNIST 数据集...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. 数据预处理：归一化到 [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 3. 构建模型（与浏览器端兼容）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4. 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. 训练模型
print("开始训练模型...")
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1, verbose=1)

# 6. 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n测试准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# 7. 保存为 SavedModel（目录格式）
saved_model_dir = "mnist_savedmodel"
tf.saved_model.save(model, saved_model_dir)
print(f"\n✅ 模型已保存为 SavedModel 格式到: {saved_model_dir}/")