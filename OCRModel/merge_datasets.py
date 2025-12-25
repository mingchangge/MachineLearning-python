import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

# ===================================================================
# --- 1. 配置 (您唯一需要修改的地方) ---
# ===================================================================

# 旧的、庞大的合成数据集路径
OLD_SYNTHETIC_DIR = "../ocr_dataset_hybrid"

# 新的、高质量的“黄金”真实数据集路径
NEW_FINETUNE_DIR = "../stratified_dataset_split"

# 输出：将要创建的最终“三明治”混合数据集的文件夹名称
OUTPUT_DIR = "../stratified_mixed_dataset"

# 【关键】您希望从旧数据集中随机抽取的样本数量
NUM_OLD_SAMPLES_TO_USE = 2750 # 推荐使用 1:5 的比例

# ===================================================================

def merge_datasets():
    """
    主函数，执行所有合并操作。
    """
    print(f"🚀 开始创建“三明治”混合数据集 '{OUTPUT_DIR}'...")

    # --- 1. 创建输出目录结构 ---
    train_dir_out = os.path.join(OUTPUT_DIR, "train")
    val_dir_out = os.path.join(OUTPUT_DIR, "val")
    train_images_out = os.path.join(train_dir_out, "images")
    val_images_out = os.path.join(val_dir_out, "images")
    
    os.makedirs(train_images_out, exist_ok=True)
    os.makedirs(val_images_out, exist_ok=True)

    # --- 2. 加载所有标签信息到Pandas DataFrame中 ---
    print("📊 正在加载标签文件...")
    old_labels_path = os.path.join(OLD_SYNTHETIC_DIR, "labels.txt")
    new_train_labels_path = os.path.join(NEW_FINETUNE_DIR, "train", "labels.txt")
    new_val_labels_path = os.path.join(NEW_FINETUNE_DIR, "val", "labels.txt")

    try:
        df_old = pd.read_csv(old_labels_path, sep='\t', header=None, names=['filepath', 'transcription'])
        df_new_train = pd.read_csv(new_train_labels_path, sep='\t', header=None, names=['filepath', 'transcription'])
        df_new_val = pd.read_csv(new_val_labels_path, sep='\t', header=None, names=['filepath', 'transcription'])
    except FileNotFoundError as e:
        print(f"❌ 错误: 找不到文件 {e.filename}。请检查您的目录配置。")
        return

    # --- 3. 【核心】处理训练集 ---
    print(f"\n⚙️ 正在处理训练集...")
    
    # 随机抽取指定数量的旧数据
    if len(df_old) < NUM_OLD_SAMPLES_TO_USE:
        print(f"⚠️ 警告：请求的旧样本数量({NUM_OLD_SAMPLES_TO_USE})大于实际数量({len(df_old)})。将使用所有旧样本。")
        df_old_sample = df_old
    else:
        df_old_sample = df_old.sample(n=NUM_OLD_SAMPLES_TO_USE, random_state=42) # random_state确保每次抽取结果都一样
    
    print(f"  - 从旧数据集中随机抽取 {len(df_old_sample)} 个样本。")
    print(f"  - 从新数据集中加载 {len(df_new_train)} 个样本。")

    # 合并新旧训练集的标签信息
    df_final_train = pd.concat([df_old_sample, df_new_train], ignore_index=True)
    print(f"  - 最终训练集总计: {len(df_final_train)} 个样本。")
    
    # 将合并后的训练集标签写入新的labels.txt
    final_train_labels_path = os.path.join(train_dir_out, "labels.txt")
    df_final_train.to_csv(final_train_labels_path, sep='\t', header=False, index=False)

    # 复制训练集图片
    print("  - 正在复制训练集图片...")
    # 复制旧图片
    for _, row in tqdm(df_old_sample.iterrows(), total=len(df_old_sample), desc="复制旧图片"):
        src = os.path.join(OLD_SYNTHETIC_DIR, row['filepath'])
        dst = os.path.join(train_images_out, os.path.basename(row['filepath']))
        if os.path.exists(src): shutil.copyfile(src, dst)
    # 复制新图片
    for _, row in tqdm(df_new_train.iterrows(), total=len(df_new_train), desc="复制新图片"):
        src = os.path.join(NEW_FINETUNE_DIR, "train", row['filepath'])
        dst = os.path.join(train_images_out, os.path.basename(row['filepath']))
        if os.path.exists(src): shutil.copyfile(src, dst)

    # --- 4. 【核心】处理验证集 (只使用新的“黄金”验证集) ---
    print(f"\n⚙️ 正在处理验证集...")
    print(f"  - 加载 {len(df_new_val)} 个“黄金”验证样本。")
    
    # 写入验证集标签
    final_val_labels_path = os.path.join(val_dir_out, "labels.txt")
    df_new_val.to_csv(final_val_labels_path, sep='\t', header=False, index=False)

    # 复制验证集图片
    print("  - 正在复制验证集图片...")
    for _, row in tqdm(df_new_val.iterrows(), total=len(df_new_val), desc="复制验证图片"):
        src = os.path.join(NEW_FINETUNE_DIR, "val", row['filepath'])
        dst = os.path.join(val_images_out, os.path.basename(row['filepath']))
        if os.path.exists(src): shutil.copyfile(src, dst)

    print(f"\n🎉🎉🎉 成功！最终的“三明治”混合数据集已在 '{OUTPUT_DIR}' 文件夹中创建。")
    print("下一步：请将这个文件夹打包成.zip，上传到Colab进行最终的微调训练。")


if __name__ == "__main__":
    merge_datasets()