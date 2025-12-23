import os
from PIL import Image # 需要安装 Pillow: pip install Pillow

LABELS_FILE = "../stratified_dataset/labels.txt"
IMAGE_ROOT = "../stratified_dataset"
error_count = 0

print("🚀 开始检查数据集的完整性和对应关系...")

with open(LABELS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        try:
            path, label = line.strip().split('\t')
            full_path = os.path.join(IMAGE_ROOT, path)
            
            # 尝试打开图片
            img = Image.open(full_path)
            
        except FileNotFoundError:
            print(f"❌ 错误！在第 {i+1} 行，找不到图片文件: {full_path}")
            error_count += 1
        except Exception as e:
            print(f"❌ 错误！在第 {i+1} 行，打开图片时发生未知错误: {e}")
            error_count += 1

if error_count == 0:
    print(f"✅ 检查完成！所有 {len(lines)} 个样本都完美对应，您的数据集非常健康！")
else:
    print(f"⚠️ 检查发现 {error_count} 个错误，请根据上面的提示进行修复。")