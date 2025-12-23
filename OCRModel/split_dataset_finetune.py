import os
import shutil
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
from collections import Counter

# ===================================================================
# --- 1. 配置 (您唯一需要修改的地方) ---
# ===================================================================

# 输入：您之前生成的、包含所有字体图片的数据集路径
INPUT_DIR = "../stratified_dataset" 

# 输出：将要生成的、划分好的新数据集的文件夹名称
OUTPUT_DIR = "../stratified_dataset_split"

# 划分比例：0.2代表 20% 的数据作为验证集，80%作为训练集
VALIDATION_SET_SIZE = 0.2

# ===================================================================

# 辅助函数，用于处理每个划分
def process_split(dataframe, split_name):
    """
    为CRNN模型创建一个划分好的数据集子目录 (train 或 val)。
    它会创建一个 images/ 文件夹和 一个总的 labels.txt 文件。
    """
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    split_image_dir = os.path.join(split_dir, "images")
    os.makedirs(split_image_dir, exist_ok=True)
    
    output_labels_path = os.path.join(split_dir, 'labels.txt')
    dataframe[['filepath', 'transcription']].to_csv(
        output_labels_path, sep='\t', header=False, index=False
    )

    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"复制 {split_name} 图片"):
        source_path = os.path.join(INPUT_DIR, row['filepath'])
        dest_path = os.path.join(split_image_dir, os.path.basename(row['filepath']))
        if os.path.exists(source_path):
            shutil.copyfile(source_path, dest_path)
        else:
            print(f"⚠️ 警告：找不到源文件 {source_path}")

def create_split_dataset():
    """
    主函数，执行所有操作。
    """
    print("🚀 开始执行数据集分层划分...")

    # 加载并解析原始标签文件 ---
    labels_path = os.path.join(INPUT_DIR, "labels.txt")
    if not os.path.exists(labels_path):
        print(f"❌ 错误: 找不到标签文件 '{labels_path}'。")
        return

    df = pd.read_csv(labels_path, sep='\t', header=None, names=['filepath', 'transcription'])
    df['font_type'] = df['filepath'].apply(lambda x: os.path.basename(x).split('_')[0])
    
    print("\n📊 原始数据集字体分布:")
    print(df['font_type'].value_counts())

    # 执行分层采样 ---
    print(f"\n⚙️ 正在按 {1-VALIDATION_SET_SIZE:.0%}/{VALIDATION_SET_SIZE:.0%} 的比例进行分层...")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_SET_SIZE, random_state=42)
    train_indices, val_indices = next(splitter.split(df, df['font_type']))
    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]
    print(f"划分完成: {len(train_df)} 训练样本, {len(val_df)} 验证样本。")
    print("\n📊 验证集字体分布 (检查均衡性):")
    print(val_df['font_type'].value_counts())
    
    # --- 创建新的目录结构并复制文件 ---
    print("\n📂 正在创建新的目录结构并复制文件...")
    process_split(train_df, "train")
    process_split(val_df, "val")
    
    print(f"\n🎉🎉🎉 成功！模型的训练/验证数据集已在 '{OUTPUT_DIR}' 文件夹中创建。")

if __name__ == "__main__":
    create_split_dataset()