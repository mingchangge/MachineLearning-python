# ===================================================================
# 专业级数据准备脚本：处理手动导出的JSON和本地文件
# ===================================================================
import os
import json
import shutil
from tqdm import tqdm


# 1. 配置 --- 【重要】请将下面四个变量修改为您自己的信息
LABEL_STUDIO_URL = "http://localhost:8080" # 您的Label Studio服务器地址
#API_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MzY3MTE1NSwiaWF0IjoxNzY2NDcxMTU1LCJqdGkiOiJiYjk4ZjQ2YzM2NGY0OWQ2YjY5MzdkOGFlZGU3ZGMzZiIsInVzZXJfaWQiOiIxIn0.qxP9ZfGv2gWDKdgKucyJxXNE7PSGQhWKyYBUClzCL4Q"            # 您的API Token (在账户设置里找)          # 您的API Token (在账户设置里找)
JSON_EXPORT_FILE = "project-1-at-2025-12-23-02-50-0a490365.json" 
LABEL_STUDIO_MEDIA_PATH = "/Users/terren/Library/Application Support/label-studio/media/upload/1"        # 您导出的JSON文件名
PROJECT_ID = 1  

# 2. 创建本地数据集目录和labels.txt 
OUTPUT_DIR = "../stratified_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)
labels_txt_path = os.path.join(OUTPUT_DIR, "labels.txt")


print("🚀 开始解析本地JSON文件并复制图片...")
with open(JSON_EXPORT_FILE, 'r') as f:
    data = json.load(f)

with open(labels_txt_path, 'w', encoding='utf-8') as labels_file:
    for task in tqdm(data, desc="处理标注任务"):
        if not task.get('annotations'): continue
        annotation = task['annotations'][0]['result']
        
        text_content = ""; font_choice = ""
        for item in annotation:
            if item.get('type') == 'textarea': text_content = item['value']['text'][0]
            elif item.get('type') == 'choices': font_choice = item['value']['choices'][0].lower()

        if not text_content or not font_choice: continue
            
        # 从JSON中获取图片的“内部路径”
        image_url_suffix = task['data']['image']
        # 提取纯粹的文件名
        original_filename = os.path.basename(image_url_suffix)
        
        # 构造源文件路径和目标文件路径
        source_image_path = os.path.join(LABEL_STUDIO_MEDIA_PATH, original_filename)
        
        # 构造新的、带字体信息的目标文件名
        new_filename = f"{font_choice}_{original_filename}"
        destination_image_path = os.path.join(IMAGES_DIR, new_filename)
        
        # 检查源文件是否存在，然后复制
        if os.path.exists(source_image_path):
            shutil.copyfile(source_image_path, destination_image_path)
            # 写入labels.txt
            labels_file.write(f"images/{new_filename}\t{text_content}\n")
        else:
            print(f"警告：找不到本地图片文件: {source_image_path}")

print(f"🎉🎉🎉 最终成功！数据集已在本地创建于 '{OUTPUT_DIR}' 文件夹中。")
print("您现在可以继续进行分层采样和模型微调了。")