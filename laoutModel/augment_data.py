import os
import cv2
import albumentations as A
from tqdm import tqdm

# 第二次训练模型数据增强
# --- 1. 配置区域 ---

# 输入目录：你的原始训练数据
INPUT_DIR = '../dataset/train'

# 输出目录：存放增强后数据的地方
OUTPUT_DIR = '../dataset_augmented_final/train'

# 为每张原始图片生成多少张增强图片
# 对于40多张的原始图片，生成15-25张增强图是比较合适的起点
NUM_AUGMENTATIONS_PER_IMAGE = 25

# --- 2. 定义对文字友好的数据增强流程 ---

transform = A.Compose([
    # 以50%的概率随机调整亮度和对比度。
    # 这是模拟不同光照环境下屏幕显示效果最安全有效的方法。
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),

    # 以40%的概率添加高斯噪声。
    # 模拟相机传感器噪声和图像压缩伪影，增强模型鲁棒性。
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),

    # 集合了平移、缩放和旋转的核心几何变换。
    # rotate_limit=5: 将旋转严格限制在±5度内，保证文字方向正确。
    # scale_limit=0.1: 缩放范围为90%-110%，模拟不同分辨率。
    # shift_limit=0.05: 平移范围为5%，模拟截图位置的轻微偏移。
    # border_mode=cv2.BORDER_CONSTANT: 旋转或平移产生的空白区域用黑色填充。
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5,
                       p=0.9, border_mode=cv2.BORDER_CONSTANT, value=0),

    # 以40%的概率应用轻微的运动模糊。
    # 极佳地模拟了拍照时手部的微小抖动，非常实用。
    A.MotionBlur(blur_limit=5, p=0.4),

], bbox_params=A.BboxParams(format='yolo',
                            label_fields=['class_labels'],
                            # 增强后，如果边界框的可见部分少于30%，则丢弃它
                            min_visibility=0.3))


# --- 3. 辅助函数 ---

def read_yolo_labels(label_path):
    """读取 YOLO 格式的标注文件。"""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_labels.append(int(parts[0]))
                bboxes.append(list(map(float, parts[1:])))
    return bboxes, class_labels

def write_yolo_labels(label_path, labels):
    """写入 YOLO 格式的标注文件。"""
    with open(label_path, 'w') as f:
        for label in labels:
            class_id, bbox = label
            # 防御性编程：确保坐标不会超出 [0.0, 1.0] 的范围
            bbox = [max(0.0, min(1.0, coord)) for coord in bbox]
            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")


# --- 4. 主执行函数 ---

def main():
    """主函数，执行数据增强流程。"""
    images_dir = os.path.join(INPUT_DIR, 'images')
    labels_dir = os.path.join(INPUT_DIR, 'labels')

    output_images_dir = os.path.join(OUTPUT_DIR, 'images')
    output_labels_dir = os.path.join(OUTPUT_DIR, 'labels')

    # 创建输出目录
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到 {len(image_files)} 张图片。将为每张图片生成 {NUM_AUGMENTATIONS_PER_IMAGE} 个增强版本...")

    # 使用 tqdm 创建一个可视化的进度条
    for img_name in tqdm(image_files, desc="增强进度"):
        img_path = os.path.join(images_dir, img_name)
        base_name, extension = os.path.splitext(img_name)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")

        # 读取原始图片和标注 (使用标准 cv2.imread 即可)
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}，已跳过。")
            continue
            
        bboxes, class_labels = read_yolo_labels(label_path)

        # 为每张图片生成 N 份增强数据
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            try:
                # 应用定义好的增强变换
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

                # 如果增强后所有标注框都因被裁切等原因消失了，则跳过此次保存
                if not augmented['bboxes']:
                    continue

                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']

                # 将类别 ID 和转换后的 bbox 重新组合
                final_labels = list(zip(aug_class_labels, aug_bboxes))

                # 构建新的文件名
                new_img_name = f"{base_name}_aug_{i}{extension}"
                new_label_name = f"{base_name}_aug_{i}.txt"
                new_img_path = os.path.join(output_images_dir, new_img_name)
                new_label_path = os.path.join(output_labels_dir, new_label_name)

                # 保存增强后的图片和标注
                cv2.imwrite(new_img_path, aug_image)
                write_yolo_labels(new_label_path, final_labels)

            except Exception as e:
                print(f"错误：在处理 {img_name} 的第 {i} 次增强时发生错误: {e}")

    print("\n数据增强完成！")
    print(f"结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()