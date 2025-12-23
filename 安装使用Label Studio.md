# 安装使用Label Studio

Label Studio 是一个开源的数据标注工具，支持多种数据类型的标注任务，如文本、图像、音频和视频等。以下是安装和使用 Label Studio 的基本步骤：

## 使用python 11 的虚拟环境安装 Label Studio
+
本地使用的是python 13版本，安装label studio 时会报错，提示有依赖不兼容，所以使用python 11 版本的虚拟环境安装。
1. 创建目录并进入：
   ```bash
   mkdir Label_Studio
   cd Label_Studio
   ```
2. 创建并激活虚拟环境：
    ```bash
    # 安装 python 11 版本 - 没有 python 11 版本，先安装 python 11 版本
    brew install python@3.11
    # 先确保你不在之前的.venv环境里 (如果终端前面有(.venv)就输入deactivate)
    deactivate
    # 创建虚拟环境
    /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv
    # 激活虚拟环境
    source venv/bin/activate
    ```
3. 安装 Label Studio：
   ```bash
   pip install -U label-studio
   ```
4. 启动 Label Studio：
   ```bash
   label-studio
   ```

## 使用 Label Studio 进行数据标注
1. 打开浏览器，访问 `http://localhost:8080` 进入 Label Studio 界面。
2. 注册账号并登录。
3. 创建一个新的标注项目，配置数据来源（如本地文件或数据库）和标注任务类型（如文本分类、实体识别等）。
    - 点击 "Create Project"。
    - 给项目命名，例如 `OCR_Finetune`。
    - 进入 "Labeling Setup" -> "Custom template"。
    - **用下面这段代码替换掉右侧的全部内容**。这是我为您写好的、专门用于OCR转录的界面配置：

        ```xml
        <View>
            <Image name="image" value="$image" zoom="true"/>
            <Header value="1. 选择这张图片的字体类型:"/>
            <Choices name="font_choice" toName="image"
                    choice="single" showInLine="true">
                <Choice value="Heiti" background="blue"/>   <!-- 黑体 -->
                <Choice value="Songti" background="green"/> <!-- 宋体 -->
                <Choice value="Xingkai" background="red"/>  <!-- 行楷 -->
            </Choices>
            
            <Header value="2. 在下方输入图片中的文字内容:"/>
            <TextArea name="transcription" toName="image" 
                rows="3"
                editable="true"
                maxSubmissions="1"
                placeholder="在此输入文本..."
            />
        </View>
        ```
    - 点击 "Save"。
4. 开始标注数据，根据任务类型进行操作。
5. 完成标注后，导出标注结果。
