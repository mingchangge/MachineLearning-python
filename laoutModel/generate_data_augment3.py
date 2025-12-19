# ===================================================================
# YOLOv8 最终战役 - 专项困难样本合成脚本
# ===================================================================
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
from tqdm import tqdm

OUTPUT_DIR = "../hard_samples_dataset"
NUM_IMAGES_TO_GENERATE = 500  # 我们要用500颗“炸弹”
VALIDATION_SPLIT = 0.1
FONT_DIR = "../fonts"
ORIGINAL_IMAGE_PATH = '../background_templates/template_pristine.jpg' # <-- 【重要】提供您原始截图的路径

# --- 2. 【关键】提取真实的蓝色背景模板 ---
original_img = Image.open(ORIGINAL_IMAGE_PATH)
# 根据您的UI，手动裁剪出那块纯净的蓝色背景区域
# 这几个坐标值 (left, top, right, bottom) 您需要自己微调
blue_bar_template = original_img.crop((10, 0, 1070, 707)) 
# --- 2. 【核心新增】: 碰撞检测函数 ---
def do_boxes_overlap(box_a, box_b, padding=10):
    """
    检查两个边界框（格式为 [x1, y1, x2, y2]）是否重叠。
    加入了padding，确保文本之间有足够的安全距离。
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    # 检查A是否在B的右边，或者B在A的右边 (考虑padding)
    if ax1 > bx2 + padding or bx1 > ax2 + padding:
        return False
    # 检查A是否在B的下边，或者B在A的下边 (考虑padding)
    if ay1 > by2 + padding or by1 > ay2 + padding:
        return False
        
    return True
# --- 2. 【终极修复】: 最严格的“亲自试菜”字体安全审查函数 ---
def is_font_truly_safe(font_path, required_chars):
    """
    通过直接获取字符的像素蒙版(getmask)，来100%确认字体是否支持该字符。
    """
    try:
        font = ImageFont.truetype(font_path, size=10)
        for char in required_chars:
            # getmask会尝试渲染字符。如果不支持，其返回的蒙版的size会是0。
            mask = font.getmask(char)
            if mask.size[0] == 0 or mask.size[1] == 0:
                # 只要有一个字符的蒙版是空的，就判定为不安全
                return False
        # 只有所有字符都能生成有效的像素蒙版，才判定为安全
        return True
    except Exception as e:
        # 如果字体文件本身有问题，直接判定为不安全
        # print(f"字体文件 {os.path.basename(font_path)} 读取失败: {e}") # 可选的调试信息
        return False
# --- 3. 【核心升级】: 执行最严格的审查 ---
print("🔍 开始对您的字体库进行最严格的“终极”安全审查...")

# 定义我们的“全科考试”内容：必须认识TARGET_WORDS里的所有单个字符
TARGET_WORDS = ["BMI", "脂肪", "体重", "0", "0%"]
REQUIRED_CHARS = "".join(list(set("".join(TARGET_WORDS)))) # 提取所有不重复的字符

all_font_paths = [os.path.join(FONT_DIR, f) for f in os.listdir(FONT_DIR) if f.lower().endswith(('.ttf', '.otf'))]
safe_font_paths = [] # 这是我们最终使用的、通过了审查的字体列表

for font_path in all_font_paths:
    font_name = os.path.basename(font_path)
    # 使用我们全新的、更严格的审查函数
    if is_font_truly_safe(font_path, REQUIRED_CHARS):
        print(f"  ✅ [通过] {font_name}")
        safe_font_paths.append(font_path)
    else:
        print(f"  ❌ [失败] {font_name} - 该字体不支持所有必需的字符，将被禁用。")

if not safe_font_paths:
    raise RuntimeError("致命错误：您的字体库中没有任何一个字体能通过最终审查！请检查您的字体文件。")

print(f"\n✅ 终极审查完成！共有 {len(safe_font_paths)} / {len(all_font_paths)} 种字体通过，将被用于数据生成。")


# --- 3. 准备目录和资源 ---
os.makedirs(os.path.join(OUTPUT_DIR, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'labels/val'), exist_ok=True)

# --- 4. 主生成循环 ---
print(f"🚀 开始制造 {NUM_IMAGES_TO_GENERATE} 个专项困难样本...")
IMG_WIDTH, IMG_HEIGHT = blue_bar_template.width, blue_bar_template.height

for i in tqdm(range(NUM_IMAGES_TO_GENERATE)):
     # 决定当前样本是进入训练集还是验证集
    split = 'val' if i % (1 / VALIDATION_SPLIT) == 0 else 'train'

    background = blue_bar_template.copy()
    draw = ImageDraw.Draw(background)
    
    # 随机选择1-2个词进行绘制
    words_to_draw = random.sample(TARGET_WORDS, random.randint(1, 2))
    drawn_boxes = [] # 记录这张图上已经画了的框
    labels_for_this_image = []

    for word in words_to_draw:
        is_position_safe = False
        max_retries = 100 # 设置最大尝试次数，防止死循环
        for _ in range(max_retries):
            # 随机生成字体和位置
            font_path = random.choice(safe_font_paths)
            print(f"DEBUG: 正在尝试绘制 -> 文字: '{word}', 字体: '{os.path.basename(font_path)}'")
            font_size = random.randint(28, 40)
            font = ImageFont.truetype(font_path, font_size)
            
            # 随机文本颜色，模拟渲染差异
            text_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
            
            # 随机位置
            x_pos = random.randint(int(IMG_WIDTH * 0.1), int(IMG_WIDTH * 0.8))
            y_pos = random.randint(int(IMG_HEIGHT * 0.2), int(IMG_HEIGHT * 0.6))
            # 计算当前尝试位置的边界框
            text_bbox = font.getbbox(word)
            current_box = [x_pos, y_pos, x_pos + text_bbox[2], y_pos + text_bbox[3]]
            
            # 检查是否与已画的框重叠
            has_collision = False
            for existing_box in drawn_boxes:
                if do_boxes_overlap(current_box, existing_box):
                    has_collision = True
                    break
            
            if not has_collision:
                is_position_safe = True
                break # 找到了一个安全的位置，跳出尝试循环
        
        # 如果找到了安全位置，就绘制并记录
        if is_position_safe:
            draw.text((x_pos, y_pos), word, font=font, fill=(255, 255, 255))
            drawn_boxes.append(current_box)
            
            # 计算YOLO标签
            x1, y1, x2, y2 = current_box
            class_id = 0
            x_center = ((x1 + x2) / 2) / IMG_WIDTH; y_center = ((y1 + y2) / 2) / IMG_HEIGHT
            width = (x2 - x1) / IMG_WIDTH; height = (y2 - y1) / IMG_HEIGHT
            labels_for_this_image.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # 保存图片和标签
    if labels_for_this_image: # 只有当成功画上了东西才保存
        img_filename = f"hard_sample_{i:04d}.png"
        background.save(os.path.join(OUTPUT_DIR, f'images/train', img_filename))
        with open(os.path.join(OUTPUT_DIR, f'labels/train', f"hard_sample_{i:04d}.txt"), 'w') as f:
            f.write("\n".join(labels_for_this_image))
            
print(f"🎉 成功生成最终的、无碰撞的专项数据集到 '{OUTPUT_DIR}'！")