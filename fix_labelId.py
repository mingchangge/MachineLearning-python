import os
import glob

def reset_labels_to_zero(labels_dir):
    """
    将指定目录下所有txt文件中id不为0的行重新设置为0
    
    Args:
        labels_dir: 标签文件目录路径
    """
    # 获取所有txt文件（排除classes.txt）
    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    txt_files = [f for f in txt_files if os.path.basename(f) != "classes.txt"]
    
    total_files = len(txt_files)
    total_modified = 0
    
    print(f"找到 {total_files} 个标签文件需要处理...")
    
    for i, txt_file in enumerate(txt_files, 1):
        try:
            # 读取文件内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            modified = False
            new_lines = []
            
            # 处理每一行
            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    new_lines.append(line)
                    continue
                
                # 分割行数据
                parts = line.split()
                if len(parts) >= 5:  # 确保有至少5个元素（类别ID + 4个坐标）
                    class_id = parts[0]
                    
                    # 如果类别ID不为0，则将其重置为0
                    if class_id != '0':
                        parts[0] = '0'
                        new_line = ' '.join(parts)
                        new_lines.append(new_line)
                        modified = True
                        total_modified += 1
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            
            # 如果有修改，写回文件
            if modified:
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines) + '\n')
                print(f"已处理 {i}/{total_files}: {os.path.basename(txt_file)}")
            
        except Exception as e:
            print(f"处理文件 {txt_file} 时出错: {e}")
    
    print(f"\n处理完成！")
    print(f"总共处理了 {total_files} 个文件")
    print(f"修改了 {total_modified} 个标签")

if __name__ == "__main__":
    # 设置标签目录路径
    labels_dir = "/Users/terren/Projects/demo/MachineLearning/dataset/val/labels"
    
    # 检查目录是否存在
    if not os.path.exists(labels_dir):
        print(f"错误: 目录 {labels_dir} 不存在")
        exit(1)
    
    # 执行重置操作
    reset_labels_to_zero(labels_dir)
