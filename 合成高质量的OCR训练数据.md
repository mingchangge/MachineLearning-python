# 训练文本识别模型

## 阶段一：合成高质量的OCR训练数据

这是整个流程中最具创造性也最关键的一步。我们的目标是编写一个Python脚本，生成成千上万张与您App截图风格类似的、带有标注的文本小图片。

### 步骤 1: 准备环境和素材

1.  **安装库**: 您需要 `Pillow` 库来创建和绘制图像。
    ```bash
    pip install Pillow
    ```

2.  **获取字体**: 这是决定合成数据质量的灵魂。
    *   **最佳选择**: 如果能通过解包App或在网上找到App使用的字体（例如 `PingFang SC`, `Helvetica Neue` 等），效果最好。
    *   **备选方案**: 使用高质量的开源中文字体。**思源黑体 (Source Han Sans / Noto Sans CJK)** 是一个绝佳的选择，因为它字形标准且覆盖字符广。请下载并将其 `.otf` 或 `.ttf` 文件放在您的项目文件夹中。

3.  **定义字符集和样式**:
    *   **字符集 (`CHARSET`)**: 仔细查看您的截图，列出所有出现过的字符。一个都不能漏！
        ```
        0123456789.% 公斤胖低标建议准高体脂率水分骨骼肌蛋白肉内脏指数皮下去身年龄型基础代谢活动重控制卡
        ```
    *   **背景颜色 (`BG_COLORS`)**: 使用颜色拾取工具从截图中提取主要背景色。
        *   标准绿: `(47, 182, 128)`
        *   偏低蓝: `(64, 169, 237)`
        *   偏高/胖橙: `(245, 166, 35)`
        *   白色/灰色: `(255, 255, 255)`, `(240, 240, 240)`
    *   **文本颜色 (`TEXT_COLORS`)**: 主要是白色 `(255, 255, 255)` 和深灰色 `(80, 80, 80)`。

### 步骤 2: 编写数据生成脚本

以下是一个完整、注释详细的Python脚本。将它保存为 `generate_ocr_data.py`。

```python
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import colorsys

# ==============================
# 1. 配置
# ==============================
OUTPUT_DIR = 'synthetic_ocr_dataset_advanced'
NUM_IMAGES_TO_GENERATE = 10000
IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50

# 字体路径（请确保该文件存在）
FONT_PATHS = [
    './fonts/vivoSansGlobal-Regular.ttf',
    './fonts/vivoSansComp400_0.ttf',
    './fonts/vivoSansComp800_0.ttf'
]  # 推荐使用思源黑体或阿里巴巴普惠体

# 字符集（来自健康类App）
CHARSET = "0123456789.%BMI对比上次测量体重公斤脂肪率水分骨骼肌蛋白质肉内脏指数皮下去身年龄型基础代谢活动建议控制偏胖高低标准肥大卡隐形微稍瘦强壮过力发达"
STATUS_WORDS = {'偏胖', '偏瘦', '标准', '偏高', '偏低', '正常', '发达', '强壮', '隐形', '微', '稍'}
# 从真实App截图中提取的背景色（绿色/蓝色/橙色/白色系）
BG_COLORS = [
    (47, 182, 128), (45, 175, 122), (50, 188, 135), (42, 155, 75), (42, 154, 74),  # 绿色系
    (64, 169, 237), (60, 162, 228), (70, 175, 242), (73, 184, 255),
    (43, 96, 128), (50, 107, 140), (47, 99, 131), (35, 85, 115),                # 蓝色系
    (239, 133, 25), (245, 166, 35), (238, 160, 30), (250, 172, 45),            # 橙色系
    (250, 250, 250), (245, 245, 245)                                            # 白色系
]

TEXT_COLORS = {
    'dark': (80, 80, 80),
    'light': (255, 255, 255),
    'blue': (68, 108, 141)
}

# ==============================
# 2. 工具函数
# ==============================

def get_font_for_text(text):
    """根据文本内容选择合适的字体"""
    if any(c.isdigit() for c in text) or any(word in text for word in STATUS_WORDS):
        # 数值或状态词 → 使用加粗字体
        return random.choice([FONT_PATHS[1], FONT_PATHS[2]])  # comp400 或 comp800
    else:
        # 字段名 → 使用常规字体
        return FONT_PATHS[0]  # global-regular

def is_value_text(text):
    """判断是否为数值或状态词"""
    if any(c.isdigit() for c in text) or any(word in text for word in STATUS_WORDS):
        return True
    return False

def choose_text_color(text, bg_color):
    """根据背景颜色和文本内容选择文字颜色"""
    # 特定背景颜色集合
    special_bg_colors = {
        (255, 165, 0),  # 橙色
        (0, 0, 255),    # 蓝色
        (0, 128, 0)     # 绿色
    }
    
    if is_dark_background(bg_color):
        return TEXT_COLORS['light']
    else:
        if is_value_text(text):
            if tuple(bg_color) in special_bg_colors:
                return TEXT_COLORS['light']  # 如果背景是特殊颜色之一，且是值或状态词，使用白色
            else:
                return TEXT_COLORS['blue']  # 否则使用蓝色
        else:
            return TEXT_COLORS['dark']  # 字段名使用黑色
        
def perturb_color_safely(rgb, sat_shift=0.06, val_shift=0.08):
    """在保持色相不变的前提下，对饱和度和明度做轻微扰动"""
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = np.clip(s + random.uniform(-sat_shift, sat_shift), 0.0, 1.0)
    v = np.clip(v + random.uniform(-val_shift, val_shift), 0.0, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

def is_dark_background(bg_color, threshold=130):
    """使用感知亮度公式判断背景是否为暗色"""
    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return luminance < threshold

# ==============================
# 3. Albumentations 增强管道（OCR友好）
# ==============================
transform = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), p=0.6),
    A.MotionBlur(blur_limit=(3, 5), p=0.3),
    A.ImageCompression(quality_lower=82, quality_upper=98, p=0.7),  # 中文需较高画质
    A.GaussNoise(var_limit=(5.0, 30.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    # 注意：移除了 GridDistortion 和 OpticalDistortion —— 对中文易造成笔画断裂
    # A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3), # 30%的概率应用轻微的网格失真，模拟屏幕变形
    # A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3), # 30%的概率应用光学畸变，模拟镜片效果
])

# ==============================
# 4. 主生成函数
# ==============================
def generate_synthetic_data_advanced():
    # 创建输出目录
    images_dir = os.path.join(OUTPUT_DIR, 'images')
    os.makedirs(images_dir, exist_ok=True)
    labels_file_path = os.path.join(OUTPUT_DIR, 'labels.txt')

    # 检查字体是否存在
    for font_path in FONT_PATHS:
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件未找到: {font_path}。请下载并放在对应路径。")

    with open(labels_file_path, 'w', encoding='utf-8') as labels_file:
        for i in range(NUM_IMAGES_TO_GENERATE):
            # 1. 生成随机文本
            text_length = random.randint(1, 8)
            text = ''.join(random.choices(CHARSET, k=text_length))

            # 2. 选择并扰动背景色
            base_bg = random.choice(BG_COLORS)
            bg_color = perturb_color_safely(base_bg)

            # 3. 根据背景亮度选择文字颜色
            text_color = choose_text_color(text, bg_color)

            # 4. 创建图像并绘制文字
            image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color=bg_color)
            draw = ImageDraw.Draw(image)
            font_size = random.randint(28, 36)
            font = ImageFont.truetype(font_path, font_size)

            # 计算文本尺寸（兼容新旧Pillow版本）
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except AttributeError:
                text_width, text_height = draw.textsize(text, font=font)

            position = ((IMAGE_WIDTH - text_width) // 2, (IMAGE_HEIGHT - text_height) // 2)
            draw.text(position, text, font=font, fill=text_color)

            # 5. 应用 Albumentations 增强
            image_np = np.array(image)
            transformed = transform(image=image_np)
            augmented_image_np = transformed['image']
            final_image = Image.fromarray(augmented_image_np)

            # 6. 保存
            image_name = f'synth_{i:06d}.png'
            image_path = os.path.join(images_dir, image_name)
            final_image.save(image_path, optimize=True)

            relative_path = os.path.join('images', image_name)
            labels_file.write(f'{relative_path}\t{text}\n')

            if (i + 1) % 500 == 0:
                print(f'✅ 已生成 {i + 1}/{NUM_IMAGES_TO_GENERATE} 张图片...')

    print(f'🎉 合成数据集生成完成！路径: {os.path.abspath(OUTPUT_DIR)}')

# ==============================
# 5. 入口
# ==============================
if __name__ == '__main__':
    generate_synthetic_data_advanced()

```

**如何使用**:
1.  将代码保存为 `generate_ocr_data.py`。
2.  下载vivo Sans（或您选择的字体），将其 `.ttf` 文件与脚本放在同一目录。
3.  运行脚本: `python generate_ocr_data.py`。
4.  运行结束后，您会得到一个 `synthetic_ocr_dataset` 文件夹，里面有一个 `images` 子文件夹和 `labels.txt` 文件，这就是我们完美的训练数据！

---

## 阶段二：选择模型架构与训练

### 模型推荐: 轻量级 CRNN + CTC Loss

这是文本识别领域的黄金标准组合。

*   **CRNN (Convolutional Recurrent Neural Network)**:
    1.  **C (Convolutional / 卷积) 部分**: 模型的“眼睛”。它由一系列卷积层组成（例如一个轻量级的MobileNet或ResNet主干），负责从输入图像中提取一系列特征图。它不关心是什么字，只负责提取笔画、边缘等视觉模式。
    2.  **R (Recurrent / 循环) 部分**: 模型的“序列处理器”。通常使用 **双向LSTM (Bi-LSTM)**。它接收来自CNN的特征序列，并学习字符之间的上下文关系（例如，“公”后面很可能跟“斤”）。
    3.  **转录层**: 一个简单的全连接层，将RNN的输出转换为每个时间步上对应所有字符的概率分布。

*   **CTC Loss (Connectionist Temporal Classification Loss)**:
    *   **核心作用**: 这是CRNN的“魔法”所在。它允许模型在**不知道每个字符在图片中确切位置**的情况下进行端到端的训练。您只需要提供一张图片和它对应的完整文本字符串，CTC Loss会自动处理对齐问题。这极大地简化了数据标注和训练过程。

### 具体训练操作 (以Kaggle Notebook为例)

训练CRNN比调用YOLO框架要复杂一些，因为它需要我们自己定义模型和训练循环。我们将使用 **PyTorch**，这是一个非常灵活且强大的框架。

1.  **准备数据**:
    *   将您生成的 `synthetic_ocr_dataset` 文件夹压缩成 `.zip`。
    *   在Kaggle上创建一个新的**私有数据集**并上传这个zip文件。

2.  **创建并设置Kaggle Notebook**:
    *   创建一个新的Notebook。
    *   在右侧面板设置 `Accelerator` 为 **GPU**，并 **开启 `Internet`**。
    *   通过 `+ Add Data` 添加您刚刚创建的合成OCR数据集。

3.  **编写训练代码 (在Notebook单元格中)**:
    这是一个完整的、可运行的PyTorch训练脚本的框架。您只需按顺序将其粘贴到Kaggle Notebook的单元格中即可。

    **Cell 1: 安装依赖与导入库**
    ```python
    # 安装一个用于计算文本编辑距离的库，方便评估模型
    !pip install python-Levenshtein -q

    import os
    import cv2
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    import numpy as np
    from collections import OrderedDict
    import zipfile
    ```

    **Cell 2: 解压数据并设置配置**
    ```python
    # 解压数据集到可写目录
    with zipfile.ZipFile('/kaggle/input/YOUR_SYNTHETIC_DATASET_NAME/synthetic_ocr_dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('/kaggle/working/')

    # --- 配置 ---
    DATA_DIR = '/kaggle/working/synthetic_ocr_dataset'
    CHARSET = "0123456789.%公斤胖低标建议准高体脂率水分骨骼肌蛋白肉内脏指数皮下去身年龄型基础代谢活动重控制卡"
    
    # 将字符映射到整数
    # 0保留给CTC的'blank' token
    char_to_int = {char: i + 1 for i, char in enumerate(CHARSET)}
    int_to_char = {i + 1: char for i, char in enumerate(CHARSET)}
    
    NUM_CLASSES = len(CHARSET) + 1 # +1 for the blank token
    IMG_WIDTH = 200
    IMG_HEIGHT = 50
    EPOCHS = 50 # 先用50轮快速迭代
    BATCH_SIZE = 64
    ```

    **Cell 3: 创建自定义数据集类**
    ```python
    class OCRDataset(Dataset):
        def __init__(self, data_dir, char_to_int_map, transform=None):
            self.data_dir = data_dir
            self.transform = transform
            self.char_to_int = char_to_int_map
            self.image_paths = []
            self.labels = []
            
            with open(os.path.join(data_dir, 'labels.txt'), 'r', encoding='utf-8') as f:
                for line in f:
                    path, label = line.strip().split('\t')
                    self.image_paths.append(os.path.join(data_dir, path))
                    self.labels.append(label)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            label = self.labels[idx]
            
            # 读取灰度图
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if self.transform:
                image = self.transform(image)
            
            # 编码文本
            encoded_label = [self.char_to_int[char] for char in label]
            
            return {'image': image, 'label': torch.IntTensor(encoded_label), 'label_length': len(encoded_label)}

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # 归一化到[-1, 1]
    ])
    ```

    **Cell 4: 定义CRNN模型架构**
    ```python
    class CRNN(nn.Module):
        def __init__(self, num_classes):
            super(CRNN, self).__init__()
            # CNN part (simplified VGG-like)
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x25x100
                nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x12x50
                nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), # 256x5x50
                nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)), # 512x2x50
                nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True) # 512x1x49
            )
            # RNN part
            self.rnn = nn.Sequential(
                nn.LSTM(512, 256, bidirectional=True),
                nn.LSTM(512, 256, bidirectional=True)
            )
            # Transcription part
            self.fc = nn.Linear(512, num_classes)

        def forward(self, x):
            # CNN forward
            conv = self.cnn(x)
            b, c, h, w = conv.size()
            assert h == 1, "the height of conv feature should be 1"
            conv = conv.squeeze(2) # [b, c, w]
            conv = conv.permute(2, 0, 1) # [w, b, c] for RNN
            
            # RNN forward
            rnn, _ = self.rnn(conv)
            
            # Transcription
            output = self.fc(rnn)
            return output
    ```

    **Cell 5: 训练循环**
    ```python
    def collate_fn(batch):
        images = [item['image'] for item in batch]
        labels = [item['label'] for item in batch]
        label_lengths = [item['label_length'] for item in batch]
        
        images = torch.stack(images, 0)
        labels = torch.cat(labels, 0)
        label_lengths = torch.IntTensor(label_lengths)
        
        return images, labels, label_lengths

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CRNN(NUM_CLASSES).to(device)
    criterion = nn.CTCLoss(blank=0, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    dataset = OCRDataset(DATA_DIR, char_to_int, transform=transform)
    # 划分训练集和验证集 80/20
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    print(f"开始训练，共 {EPOCHS} 轮...")
    for epoch in range(EPOCHS):
        model.train()
        for i, (images, labels, label_lengths) in enumerate(train_loader):
            images = images.to(device)
            
            preds = model(images) # [seq_len, batch_size, num_classes]
            preds_size = torch.IntTensor([preds.size(0)] * images.size(0))
            
            loss = criterion(preds.log_softmax(2), labels, preds_size, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 50 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    print("训练完成！")

    # 保存模型
    torch.save(model.state_dict(), '/kaggle/working/crnn_model.pth')
    print("模型已保存到 /kaggle/working/crnn_model.pth")
    ```

4.  **保存并获取模型**:
    *   点击右上角的 `Save Version` -> `Save & Run All (Commit)`。
    *   等待Kaggle在后台运行完毕。
    *   回到Notebook的 `Data` -> `Output` 页面，您就可以找到并下载训练好的 `crnn_model.pth` 文件了。

## 阶段三：模型转换 (PyTorch -> ONNX -> TF.js)

由于浏览器环境原生支持TF.js，我们需要将PyTorch模型转换。

1.  **在Notebook中增加导出代码**:
    在训练代码下方添加新的单元格，用于将`.pth`文件转换为通用格式 **ONNX**。

    ```python
    # 首先加载我们训练好的模型
    model.load_state_dict(torch.load('/kaggle/working/crnn_model.pth'))
    model.eval()

    # 创建一个虚拟输入，尺寸要和模型输入一致
    dummy_input = torch.randn(1, 1, IMG_HEIGHT, IMG_WIDTH).to(device)
    
    torch.onnx.export(model,
                      dummy_input,
                      "/kaggle/working/crnn_model.onnx",
                      export_params=True,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {1: 'batch_size'}})

    print("模型已导出为 ONNX 格式: /kaggle/working/crnn_model.onnx")
    ```

2.  **安装ONNX到TF.js转换工具并转换**:

    ```bash
    !pip install onnx onnx-tf -q
    !onnx-tf convert -i /kaggle/working/crnn_model.onnx -o /kaggle/working/tfjs_model
    ```
    * 这条命令会将 `.onnx` 文件转换为一个包含 `model.json` 和权重文件的 `tfjs_model` 文件夹。

3.  **最终保存**:
    *   再次 `Save Version` -> `Save & Run All (Commit)`。
    *   完成后，在 `Output` 中，您就可以下载最终的 `tfjs_model` 文件夹了。这就是您的第二个轻量化模型！

这个流程虽然步骤繁多，但每一步都是构建一个可控、高性能自定义AI模型的标准实践。您现在拥有了从数据创造到模型部署的全套技能。