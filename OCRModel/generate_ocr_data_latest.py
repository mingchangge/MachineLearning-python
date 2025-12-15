import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import albumentations as A
import colorsys
import shutil
# 第二次训练文字识别模型最终版
# ==============================
# 1. 配置 (Configuration)
# ==============================
# --- 基本设置 ---
OUTPUT_DIR = '../ocr_dataset_hybrid'
FONTS_DIR = "../fonts"
NUM_IMAGES_TO_GENERATE = 30000  # 建议生成至少 10,000 张以获得良好效果
IMAGE_WIDTH = 256  # 增加宽度以容纳更长的文本和几何变换
IMAGE_HEIGHT = 64  # 增加高度

# --- 文本内容模板 (核心优化) ---
CHARSET = "0123456789.%BMI对比上次测量体重公斤脂肪率水分骨骼肌蛋白质肉内脏指数皮下去身年龄型基础代谢活动建议控制偏胖高低标准肥大卡隐形微稍瘦强壮过力发达%()-:（）：-日期健康弱"
VALUE_TEMPLATES = ["{:.1f}", "{:.2f}", "{}", "{:.1f}%"]
LABEL_TEMPLATES = ["体重", "BMI", "体脂率", "水分", "骨骼肌", "蛋白质", "内脏脂肪指数", "身体年龄", "基础代谢", "去脂体重", "皮下脂肪"]
STATUS_TEMPLATES = ["偏胖", "标准", "偏瘦", "正常", "偏高", "偏低", "强壮", "发达", "肥胖型", "肌肉型", "健康"]
UNIT_TEMPLATES = ["公斤", "大卡", "%"]

# --- 视觉样式 ---
BG_COLORS = [
    (47, 182, 128), (45, 175, 122), (50, 188, 135),  # 绿色系
    (64, 169, 237), (60, 162, 228), (70, 175, 242),  # 蓝色系
    (239, 133, 25), (245, 166, 35), (238, 160, 30),  # 橙色系
    (250, 250, 250), (245, 245, 245)                 # 白色系
]
TEXT_COLORS = {
    'dark': (80, 80, 80),
    'light': (255, 255, 255),
    'blue': (68, 108, 141) # App中数值在白色背景下的颜色
}

# --- 字体资源 ---
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
FONT_PATHS = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith(('.ttf', '.otf'))]

if not FONT_PATHS:
    raise FileNotFoundError(f"在 '{FONTS_DIR}' 目录中未找到任何字体文件。请确保字体文件存在。")
print(f"✅ 成功加载了 {len(FONT_PATHS)} 种字体。")

# ==============================
# 2. Albumentations 增强管道 (Augmentation Pipeline)
# ==============================
transform = A.Compose([
    # --- 强度和模糊 ---
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.7),
        A.MotionBlur(blur_limit=(3, 7), p=0.7),
    ], p=0.8), # 80%的概率应用模糊

    # --- 噪声和压缩伪影 ---
    A.ImageCompression(quality_lower=75, quality_upper=95, p=0.8),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),

    # --- 几何变换 (对OCR鲁棒性至关重要) ---
    A.ShiftScaleRotate(
        shift_limit=0.06,      # 最多平移6%
        scale_limit=0.1,       # 最多缩放10%
        rotate_limit=2.5,      # 最多旋转±2.5度
        border_mode=cv2.BORDER_CONSTANT,
        p=0.8
    ),
    A.Perspective(scale=(0.02, 0.05), p=0.5),
])

# ==============================
# 3. 工具函数 (Utility Functions)
# ==============================

def generate_structured_text():
    """【核心】生成更真实的、有结构的文本，而非随机字符"""
    category = random.choices(['value', 'label', 'status', 'value_with_unit'], weights=[4, 3, 2, 2], k=1)[0]
    
    if category == 'value':
        template = random.choice(VALUE_TEMPLATES)
        if "{}" in template: return template.format(random.randint(20, 2000))
        else: return template.format(random.uniform(10.0, 100.0))
            
    if category == 'label': return random.choice(LABEL_TEMPLATES)
    if category == 'status': return random.choice(STATUS_TEMPLATES)
        
    if category == 'value_with_unit':
        val_template = random.choice(VALUE_TEMPLATES[:2])
        value = val_template.format(random.uniform(10.0, 100.0))
        unit = random.choice(UNIT_TEMPLATES)
        return f"{value} {unit}" # 模拟中间有空格的情况


def is_dark_background(bg_color, threshold=130):
    """使用感知亮度公式判断背景是否为暗色"""
    luminance = 0.299 * bg_color[0] + 0.587 * bg_color[1] + 0.114 * bg_color[2]
    return luminance < threshold

def choose_text_color(text, base_bg_color):
    """根据背景颜色和文本内容选择文字颜色"""
    if is_dark_background(base_bg_color):
        return TEXT_COLORS['light']
    else: # Light background
        if any(c.isdigit() for c in text):
            return TEXT_COLORS['blue']
        else:
            return TEXT_COLORS['dark']

def perturb_color(rgb):
    """对颜色进行轻微扰动"""
    r, g, b = [x / 255.0 for x in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s = np.clip(s + random.uniform(-0.08, 0.08), 0.0, 1.0)
    v = np.clip(v + random.uniform(-0.1, 0.1), 0.0, 1.0)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

def create_gradient_background(color1, color2, width, height):
    """创建从上到下的线性渐变背景"""
    base = Image.new('RGB', (width, height), color1)
    top = Image.new('RGB', (width, height), color2)
    mask = Image.new('L', (width, height))
    mask_data = np.array(mask)
    mask_data[:, :] = np.linspace(255, 0, height)[:, np.newaxis]
    mask = Image.fromarray(mask_data)
    base.paste(top, (0, 0), mask)
    return base

# ==============================
# 4. 主生成函数 (Main Generation Function)
# ==============================
def generate_synthetic_data_final():
    """主函数，负责生成整个数据集"""
    # --- 初始化 ---
    if os.path.exists(OUTPUT_DIR):
        print(f"警告：输出目录 {OUTPUT_DIR} 已存在，将进行覆盖。")
        shutil.rmtree(OUTPUT_DIR)
        
    images_dir = os.path.join(OUTPUT_DIR, 'images')
    os.makedirs(images_dir, exist_ok=True)
    labels_file_path = os.path.join(OUTPUT_DIR, 'labels.txt')

    for font_path in FONT_PATHS:
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件未找到: {font_path}。请确保'fonts'目录和其中的字体文件存在。")

    print("🚀 开始生成高级合成OCR数据集...")
    with open(labels_file_path, 'w', encoding='utf-8') as labels_file:
        for i in range(NUM_IMAGES_TO_GENERATE):
            # 1. 生成结构化文本
            text = generate_structured_text()

            # 2. 确定样式（颜色，字体）
            base_bg_color = random.choice(BG_COLORS)
            text_color = choose_text_color(text, base_bg_color)
            font_path = random.choice(FONT_PATHS) 
            font_size = random.randint(32, 40)
            font = ImageFont.truetype(font_path, font_size)
            
            # 3. 创建背景（加入渐变和扰动）
            bg_color_1 = perturb_color(base_bg_color)
            bg_color_2 = perturb_color(base_bg_color)
            image = create_gradient_background(bg_color_1, bg_color_2, IMAGE_WIDTH, IMAGE_HEIGHT)
            draw = ImageDraw.Draw(image)

            # 4. 绘制文本（加入位置随机性）
            try: bbox = draw.textbbox((0, 0), text, font=font)
            except AttributeError: bbox = (0, 0) + draw.textsize(text, font=font)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

            safe_margin_x = (IMAGE_WIDTH - text_width) // 2
            safe_margin_y = (IMAGE_HEIGHT - text_height) // 2
            
            if safe_margin_x > 10 and safe_margin_y > 5:
                pos_x = random.randint(int(safe_margin_x * 0.8), int(safe_margin_x * 1.2))
                pos_y = random.randint(int(safe_margin_y * 0.8), int(safe_margin_y * 1.2))
                draw.text((pos_x, pos_y), text, font=font, fill=text_color)
            else: # 如果文本太长，就居中放置
                draw.text(((IMAGE_WIDTH - text_width) // 2, (IMAGE_HEIGHT - text_height) // 2), text, font=font, fill=text_color)
                
            # 5. 应用强大的Albumentations增强
            image_np = np.array(image)
            # 动态设置 border_mode 的填充颜色为背景色，效果更佳
            transform.transforms[3].border_mode = cv2.BORDER_CONSTANT
            transform.transforms[3].value = bg_color_1 
            # Perspective变换同样需要设置
            transform.transforms[4].border_mode = cv2.BORDER_CONSTANT
            transform.transforms[4].value = bg_color_1

            augmented_image_np = transform(image=image_np)['image']
            final_image = Image.fromarray(augmented_image_np)

            # 6. 保存图像和标签
            image_name = f'synth_{i:06d}.png'
            image_path = os.path.join(images_dir, image_name)
            final_image.save(image_path, quality=95)

            relative_path = os.path.join('images', image_name)
            labels_file.write(f'{relative_path}\t{text}\n')

            if (i + 1) % 500 == 0:
                print(f'✅ 已生成 {i + 1}/{NUM_IMAGES_TO_GENERATE} 张图片...')

    print(f'\n🎉 数据集生成完毕！路径: {os.path.abspath(OUTPUT_DIR)}')
    print(f"    共生成 {NUM_IMAGES_TO_GENERATE} 张图片及其标签。")

# ==============================
# 5. 执行入口 (Execution Entry Point)
# ==============================
if __name__ == '__main__':
    generate_synthetic_data_final()