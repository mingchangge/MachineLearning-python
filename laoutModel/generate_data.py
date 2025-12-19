# 生成微调数据集的脚本
import os
import random
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm


OUTPUT_DIR = "../finetune_layout_dataset"
NUM_IMAGES_TO_GENERATE = 500  # 生成500个高质量样本
FONT_DIR = "../fonts"
TEMPLATE_DIR = "../background_templates"



IMG_WIDTH, IMG_HEIGHT = 640, 640

# --- 2. 准备目录 ---
os.makedirs(os.path.join(OUTPUT_DIR, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/val'), exist_ok=True)

font_paths = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.endswith(('.ttf', '.otf'))]
template_paths = [os.path.join(TEMPLATE_DIR, f) for f in os.listdir(TEMPLATE_DIR)]

# --- 3. 【核心升级】定义带权重的“样本类别”列表 ---
# 我们在这里精确控制每种样本的生成概率
# 'weight'越高的类别，出现的频率越高
SAMPLE_CATEGORIES = [
    # --- 靶向强化区 (高权重) ---
    {
        "name": "critical_zero",
        "text_options": ["0"],
        "weight": 50, # 权重最高，确保'0'被大量练习
        "x_range": (0.45, 0.55), "y_range": (0.28, 0.32),
        "size_range": (35, 45), "color": (255, 255, 255, 255)
    },
    {
        "name": "important_bmi",
        "text_options": ["BMI"],
        "weight": 30, # 权重次之，重点解决'BMI'的关联问题
        "x_range": (0.45, 0.55), "y_range": (0.33, 0.37),
        "size_range": (20, 30), "color": (255, 255, 255, 255)
    },
    # --- 知识巩固区 (低权重) ---
    {
        "name": "refresher_negative_float",
        "text_options": ["-5.5", "-0.2", "-8.9", "-11.2"],
        "weight": 10, # 权重较低，作为复习
        "x_range": (0.45, 0.55), "y_range": (0.28, 0.32),
        "size_range": (35, 45), "color": (255, 255, 255, 255)
    },
    {
        "name": "refresher_positive_float",
        "text_options": ["5.5", "8.9", "61.7", "36.9%"],
        "weight": 10, # 权重较低，作为复习
        "x_range": (0.4, 0.5), "y_range": (0.45, 0.85),
        "size_range": (30, 40), "color": (80, 80, 80, 255)
    }
]

# 提取权重列表，用于加权随机抽样
category_weights = [cat["weight"] for cat in SAMPLE_CATEGORIES]

# --- 4. 主生成循环 ---
print(f"🚀 开始生成 {NUM_IMAGES_TO_GENERATE} 套微调数据...")

for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
    is_val = i % 10 == 0
    split = 'val' if is_val else 'train'
    
    template_path = random.choice(template_paths)
    background = Image.open(template_path).convert("RGBA").resize((IMG_WIDTH, IMG_HEIGHT))
    
    labels_for_this_image = []

    # 随机决定在这张图上画几个样本
    # --- 【核心修改】使用带权重的随机抽样 ---
    # 随机决定在这张图上画1个还是2个样本
    num_samples_to_draw = random.randint(1, 2)
    # 按照上面定义的权重，来抽取要生成的样本类别
    categories_to_draw = random.choices(SAMPLE_CATEGORIES, weights=category_weights, k=num_samples_to_draw)


    for area_info in categories_to_draw:
        text_to_draw = random.choice(area_info["text_options"])
        font_path = random.choice(font_paths)
        font_size = random.randint(*area_info["size_range"])
        font = ImageFont.truetype(font_path, font_size)
        
        # 随机位置
        x_pos = int(random.uniform(*area_info["x_range"]) * IMG_WIDTH)
        y_pos = int(random.uniform(*area_info["y_range"]) * IMG_HEIGHT)
        
        # 绘制
        draw = ImageDraw.Draw(background)
        draw.text((x_pos, y_pos), text_to_draw, font=font, fill=area_info["color"])
        
        # 计算YOLO BBox
        try: # 使用textbbox来获得更准确的边界
            bbox = draw.textbbox((x_pos, y_pos), text_to_draw, font=font)
            x1, y1, x2, y2 = bbox
        except AttributeError: # 兼容旧版Pillow
            text_width, text_height = draw.textsize(text_to_draw, font=font)
            x1, y1 = x_pos, y_pos
            x2, y2 = x_pos + text_width, y_pos + text_height

        # 转换为YOLO格式
        class_id = 0
        x_center = ((x1 + x2) / 2) / IMG_WIDTH
        y_center = ((y1 + y2) / 2) / IMG_HEIGHT
        width = (x2 - x1) / IMG_WIDTH
        height = (y2 - y1) / IMG_HEIGHT
        
        labels_for_this_image.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存图片和标签
    img_filename = f"finetune_sample_{i:04d}.png"
    background.convert("RGB").save(os.path.join(OUTPUT_DIR, f'images/{split}', img_filename))
    
    label_filename = f"finetune_sample_{i:04d}.txt"
    with open(os.path.join(OUTPUT_DIR, f'labels/{split}', label_filename), 'w') as f:
        f.write("\n".join(labels_for_this_image))

print(f"🎉 成功生成微调数据集到 '{OUTPUT_DIR}' 文件夹！")