import os
import random
import shutil

# 随机生成数据集和验证集
# --- 配置区 ---
BASE_DIR = "../dataset"
# 支持的图片扩展名，可以根据需要添加
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png'] 

# --- 脚本正文 ---
images_dir = os.path.join(BASE_DIR, "images")
labels_dir = os.path.join(BASE_DIR, "labels")

# 创建目标文件夹
train_img_dir = os.path.join(BASE_DIR, "train/images")
val_img_dir = os.path.join(BASE_DIR, "val/images")
train_lbl_dir = os.path.join(BASE_DIR, "train/labels")
val_lbl_dir = os.path.join(BASE_DIR, "val/labels")

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(train_lbl_dir, exist_ok=True)
os.makedirs(val_lbl_dir, exist_ok=True)

# 1. 获取所有合法的图片文件名（包含扩展名），并过滤隐藏文件
all_image_files = [
    f for f in os.listdir(images_dir) 
    if not f.startswith('.') and any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS)
]
random.shuffle(all_image_files) # 随机打乱

# 2. 按比例划分
split_ratio = 0.8
split_index = int(len(all_image_files) * split_ratio)
train_files = all_image_files[:split_index]
val_files = all_image_files[split_index:]

# 3. 定义健壮的文件移动函数
def move_files(files, source_img, source_lbl, dest_img, dest_lbl):
    moved_count = 0
    for img_filename in files:
        # 从图片全名中获取不带扩展名的基本名
        base_name = os.path.splitext(img_filename)[0]
        label_filename = base_name + ".txt"

        source_img_path = os.path.join(source_img, img_filename)
        source_lbl_path = os.path.join(source_lbl, label_filename)

        # 移动前检查文件是否存在，避免崩溃
        if os.path.exists(source_img_path) and os.path.exists(source_lbl_path):
            shutil.move(source_img_path, dest_img)
            shutil.move(source_lbl_path, dest_lbl)
            moved_count += 1
        else:
            print(f"警告：找不到文件对，跳过 -> 图片: {img_filename} 或 标签: {label_filename}")
    return moved_count

# 4. 执行移动
print("开始移动训练集文件...")
moved_train = move_files(train_files, images_dir, labels_dir, train_img_dir, train_lbl_dir)

print("开始移动验证集文件...")
moved_val = move_files(val_files, images_dir, labels_dir, val_img_dir, val_lbl_dir)

print("\n--- 数据划分完成! ---")
print(f"总图片数: {len(all_image_files)}")
print(f"计划训练集: {len(train_files)} -> 成功移动: {moved_train}")
print(f"计划验证集: {len(val_files)} -> 成功移动: {moved_val}")

# 检查是否有文件遗漏
remaining_images = [f for f in os.listdir(images_dir) if not f.startswith('.')]
if remaining_images:
    print(f"\n警告：原始 images 文件夹中仍有 {len(remaining_images)} 个文件遗留！")
    print("可能的原因是它们的标签文件(.txt)不存在或名称不匹配。")


# 确定没有问题后，可以删除原始 images 和 labels 文件夹
# shutil.rmtree(images_dir)
# shutil.rmtree(labels_dir)