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