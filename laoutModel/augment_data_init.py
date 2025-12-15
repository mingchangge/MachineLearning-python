# 数据增强

import os
import cv2
import albumentations as A
import numpy as np

# 第一次训练模型数据增强-最初版
# --- 配置 ---
INPUT_DIR = '../dataset'
OUTPUT_DIR = '../dataset_augmented'
# 您希望为每张原始图片生成多少张增强图片
IMAGES_PER_SOURCE = 80

# --- 1. 定义我们的数据增强管道 (Compose) ---
# Compose会将多种增强技术组合在一起，并按顺序随机应用
transform = A.Compose([
    # 旋转 (-3到3度), 50%概率
    A.Rotate(limit=3, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    
    # 亮度/对比度调整 (-25%到+25%), 50%概率
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),

    # 色彩抖动, 30%概率
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    
    # 添加高斯噪声, 20%概率
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
    
    # 水平翻转 (可能不适用于OCR/布局任务，但作为示例)
    # A.HorizontalFlip(p=0.5),

    # 随机裁剪并缩放回原尺寸
    A.RandomSizedBBoxSafeCrop(width=480, height=800, erosion_rate=0.1, p=0.3),

], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
# bbox_params 是关键！它告诉Albumentations如何处理边界框

def augment_and_save():
    # 创建输出目录
    output_images_dir = os.path.join(OUTPUT_DIR, 'images')
    output_labels_dir = os.path.join(OUTPUT_DIR, 'labels')
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    images_dir = os.path.join(INPUT_DIR, 'images')
    labels_dir = os.path.join(INPUT_DIR, 'labels')

    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    total_images = len(image_files)
    current_image = 0
    
    for image_name in image_files:
        current_image += 1
        print(f"处理图片 {current_image}/{total_images}: {image_name}")

        image_path = os.path.join(images_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_name)

        # 读取图片
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations 使用 RGB

        # 读取YOLO格式的标注
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    bboxes.append(coords)
                    class_labels.append(class_id)
        
        # --- 循环生成增强图片 ---
        for i in range(IMAGES_PER_SOURCE):
            try:
                # 应用增强
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_class_labels = transformed['class_labels']

                # 如果增强后没有边界框了 (例如被完全裁掉了)，就跳过
                if not transformed_bboxes:
                    continue

                # --- 保存增强后的图片和标注 ---
                new_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.jpg"
                new_label_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.txt"
                
                output_image_path = os.path.join(output_images_dir, new_image_name)
                output_label_path = os.path.join(output_labels_dir, new_label_name)

                # 保存图片
                cv2.imwrite(output_image_path, cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

                # 保存新的YOLO标注
                with open(output_label_path, 'w') as f:
                    for bbox, class_id in zip(transformed_bboxes, transformed_class_labels):
                        # YOLO格式: class_id x_center y_center width height
                        line = f"{class_id} {' '.join(map(str, bbox))}\n"
                        f.write(line)
            except Exception as e:
                print(f"警告：在增强 {image_name} 的第 {i} 次时发生错误: {e}")


if __name__ == '__main__':
    augment_and_save()
    print("数据增强完成！")