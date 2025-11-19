## 安装多版本python，在vscode中使用虚拟环境

- 创建虚拟环境，python版本：3.13.11

```bash
python3 -m venv .venv
```
- 激活虚拟环境：

    - 在 macOS/Linux 上：
    ```bash
    source .venv/bin/activate
    ```
    - 在 Windows 上：
    ```bash
    .venv\Scripts\activate
    ```
后续就可以直接在vscode中使用这个虚拟环境了。

- 安装依赖命令：
```bash
pip install tensorflow
```
- 运行文件
```bash
python neuralNetworks2.py
```

# 使用 Python 训练 MNIST 模型并转换为 JSON

💡 **<font color='red'>重要提示： </font>** 不想被兼容搞的焦头烂额，使用稳定版python。

- 安装工具包命令：`pip install tensorflow tensorflowjs`

- 运行训练脚本命令：`python train_mnist_tfjs.py`

- 最终结果：测试准确率: 0.9698 (96.98%)，但保存模型失败！放弃使用 Python 训练模型，改用 node.js 训练。

# 将 MNIST 数据集转换为 JSON 格式

- 运行转换脚本命令：`python convert_mnist_to_json.py`

- 结果文件：`mnist_data.json`