# ===================================================================
# YOLOv8 布局模型微调 - 最终版、最全困难样本靶向增强脚本
# ===================================================================

import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# --- 1. 配置 ---
OUTPUT_DIR = "../finetune_augment_dataset1"
NUM_IMAGES_TO_GENERATE = 2000
FONT_DIR = "../fonts"
TEMPLATE_DIR = "../background_templates"

IMG_WIDTH, IMG_HEIGHT = 640, 640

# --- 2. 准备目录和资源 ---
os.makedirs(os.path.join(OUTPUT_DIR, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/val'), exist_ok=True)

font_paths = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]
template_paths = [os.path.join(TEMPLATE_DIR, f) for f in os.listdir(TEMPLATE_DIR)]

# --- 3. 【核心升级】定义完整的困难样本列表 ---
MINORITY_SAMPLES = [
    {
        "name": "isolated_zero_on_blue",
        "text_options": ["0", "0.0"],
        "weight": 40, # 权重最高，集中火力
        "x_range": (0.45, 0.55), "y_range": (0.28, 0.32),
        "size_range": (35, 45), "color": (255, 255, 255, 255),
        "template_keyword": "blue"
    },
    {
        "name": "isolated_number_on_white",
        "text_options": ["34", "35", "28", "41"],
        "weight": 25, # 权重次之
        "x_range": (0.75, 0.85), "y_range": (0.73, 0.77),
        "size_range": (30, 40), "color": (80, 80, 80, 255),
        "template_keyword": "white"
    },
    {
        "name": "short_label_bmi",
        "text_options": ["BMI"],
        "weight": 20, # 给予足够权重
        "x_range": (0.45, 0.55), "y_range": (0.33, 0.37),
        "size_range": (20, 30), "color": (255, 255, 255, 255),
        "template_keyword": "blue"
    },
    {
        "name": "short_label_fat",
        "text_options": ["脂肪"],
        "weight": 15, # 给予一定权重
        "x_range": (0.75, 0.85), "y_range": (0.33, 0.37), # '脂肪'标签在右侧
        "size_range": (20, 30), "color": (255, 255, 255, 255),
        "template_keyword": "blue"
    }
]

sample_weights = [s["weight"] for s in MINORITY_SAMPLES]
blue_templates = [p for p in template_paths if 'blue' in os.path.basename(p).lower() or 'main' in os.path.basename(p).lower()] or template_paths
white_templates = [p for p in template_paths if 'white' in os.path.basename(p).lower() or 'main' in os.path.basename(p).lower()] or template_paths

# --- 4. 主生成循环 ---
print(f"🚀 开始生成 {NUM_IMAGES_TO_GENERATE} 套'全明星'困难样本微调数据...")

for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
    is_val = i % 10 == 0
    split = 'val' if is_val else 'train'
    
    # 为了简化，我们随机选择一个模板作为基础
    template_path = random.choice(template_paths)
    background = Image.open(template_path).convert("RGBA").resize((IMG_WIDTH, IMG_HEIGHT))
    
    labels_for_this_image = []

    # 随机决定在这张图上画几个样本 (1到3个)
    num_samples_to_draw = random.randint(1, 3)
    # 按照权重，随机抽取要生成的样本类型
    samples_to_draw = random.choices(MINORITY_SAMPLES, weights=sample_weights, k=num_samples_to_draw)

    for sample_type in samples_to_draw:
        text_to_draw = random.choice(sample_type["text_options"])
        font_path = random.choice(font_paths)
        font_size = random.randint(*sample_type["size_range"])
        font = ImageFont.truetype(font_path, font_size)
        
        # 随机位置
        x_pos = int(random.uniform(*sample_type["x_range"]) * IMG_WIDTH)
        y_pos = int(random.uniform(*sample_type["y_range"]) * IMG_HEIGHT)
        
        # 绘制
        draw = ImageDraw.Draw(background)
        draw.text((x_pos, y_pos), text_to_draw, font=font, fill=sample_type["color"])
        
        # 计算YOLO BBox
        try:
            bbox = draw.textbbox((x_pos, y_pos), text_to_draw, font=font)
            x1, y1, x2, y2 = bbox
        except AttributeError:
            text_width, text_height = draw.textsize(text_to_draw, font=font)
            x1, y1, x2, y2 = x_pos, y_pos, x_pos + text_width, y_pos + text_height

        # 转换为YOLO格式
        class_id = 0
        x_center = ((x1 + x2) / 2) / IMG_WIDTH
        y_center = ((y1 + y2) / 2) / IMG_HEIGHT
        width = (x2 - x1) / IMG_WIDTH
        height = (y2 - y1) / IMG_HEIGHT
        
        labels_for_this_image.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存图片和标签
    img_filename = f"augment_sample_{i:04d}.png"
    background.convert("RGB").save(os.path.join(OUTPUT_DIR, f'images/{split}', img_filename))
    
    label_filename = f"augment_sample_{i:04d}.txt"
    with open(os.path.join(OUTPUT_DIR, f'labels/{split}', label_filename), 'w') as f:
        f.write("\n".join(labels_for_this_image))

print(f"🎉 成功生成'全明星'困难样本数据集到 '{OUTPUT_DIR}' 文件夹！")