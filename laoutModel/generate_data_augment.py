# ===================================================================
# YOLOv8 布局模型微调 - “少数派报告”数据合成脚本 (最终修复版)
# ===================================================================

import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

# --- 1. 配置 ---
# 请将脚本放置在您的项目根目录，确保相对路径正确
OUTPUT_DIR = "../finetune_dataset_augment"
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

# --- 3. 定义“少数派”样本列表 ---
MINORITY_SAMPLES = [
    {
        "name": "isolated_zero_on_blue",
        "text_options": ["0", "0.0"],
        "weight": 60,
        "x_range": (0.45, 0.55), "y_range": (0.28, 0.32),
        "size_range": (35, 45), "color": (255, 255, 255, 255),
        "template_keyword": "blue"
    },
    {
        "name": "isolated_number_on_white",
        "text_options": ["34", "35", "28", "41"],
        "weight": 40,
        "x_range": (0.75, 0.85), "y_range": (0.73, 0.77),
        "size_range": (30, 40), "color": (80, 80, 80, 255),
        "template_keyword": "white"
    }
]

sample_weights = [s["weight"] for s in MINORITY_SAMPLES]
# 筛选出特定模板，如果找不到，就使用所有模板
blue_templates = [p for p in template_paths if 'blue' in os.path.basename(p).lower() or 'main' in os.path.basename(p).lower()] or template_paths
white_templates = [p for p in template_paths if 'white' in os.path.basename(p).lower() or 'main' in os.path.basename(p).lower()] or template_paths

# --- 4. 主生成循环 ---
print(f"🚀 开始生成 {NUM_IMAGES_TO_GENERATE} 套'少数派报告'微调数据...")

for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
    is_val = i % 10 == 0
    split = 'val' if is_val else 'train'
    
    # 【修复点】: 逻辑简化，每张图只生成一个困难样本，直接从 MINORITY_SAMPLES 中抽样
    sample_type = random.choices(MINORITY_SAMPLES, weights=sample_weights, k=1)[0]
    
    # 根据样本类型选择合适的背景模板
    if sample_type["template_keyword"] == "blue":
        template_path = random.choice(blue_templates)
    elif sample_type["template_keyword"] == "white":
        template_path = random.choice(white_templates)
    else:
        template_path = random.choice(template_paths)

    background = Image.open(template_path).convert("RGBA").resize((IMG_WIDTH, IMG_HEIGHT))
    
    # 【修复点】: 删除了多余的内层循环，直接使用抽出的 sample_type
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
        x1, y1 = x_pos, y_pos
        x2, y2 = x_pos + text_width, y_pos + text_height

    # 转换为YOLO格式
    class_id = 0
    x_center = ((x1 + x2) / 2) / IMG_WIDTH
    y_center = ((y1 + y2) / 2) / IMG_HEIGHT
    width = (x2 - x1) / IMG_WIDTH
    height = (y2 - y1) / IMG_HEIGHT
    
    # 【修复点】: 定义一个简单的字符串来存储这一行的标签
    label_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # 保存图片和标签
    img_filename = f"minority_sample_{i:04d}.png"
    background.convert("RGB").save(os.path.join(OUTPUT_DIR, f'images/{split}', img_filename))
    
    label_filename = f"minority_sample_{i:04d}.txt"
    with open(os.path.join(OUTPUT_DIR, f'labels/{split}', label_filename), 'w') as f:
        f.write(label_line)

print(f"🎉 成功生成'少数派报告'数据集到 '{OUTPUT_DIR}' 文件夹！")