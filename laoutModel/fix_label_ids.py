#!/usr/bin/env python3
"""
修复标签文件中的ID错误
将标签文件中的ID 15改为0，16改为1
"""

import os
import glob

def fix_label_ids(label_dir):
    """
    修复标签目录下所有txt文件中的ID错误
    
    Args:
        label_dir: 标签文件所在目录
    """
    # 获取目录下所有txt文件，但排除classes.txt
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    label_files = [f for f in label_files if os.path.basename(f) != "classes.txt"]
    
    for file_path in label_files:
        print(f"处理文件: {file_path}")
        
        # 读取原始内容
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 处理每一行
        modified_lines = []
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
            
            # 分割行内容
            parts = line.split()
            if not parts:
                continue
            
            # 修改ID
            if parts[0] == '15':
                parts[0] = '0'
            elif parts[0] == '16':
                parts[0] = '1'
            
            # 重新组合行
            modified_line = ' '.join(parts)
            modified_lines.append(modified_line)
        
        # 写回文件
        with open(file_path, 'w') as f:
            f.write('\n'.join(modified_lines) + '\n')
    
    print("所有标签文件ID修复完成！")

if __name__ == "__main__":
    # 标签文件目录路径
    label_directory = "../dataset/labels"
    fix_label_ids(label_directory)