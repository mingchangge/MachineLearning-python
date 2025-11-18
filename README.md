## 安装多版本python，在vscode中使用虚拟环境

- 创建虚拟环境

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