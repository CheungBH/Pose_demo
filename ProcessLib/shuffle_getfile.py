import os
import random
import shutil
from tqdm import tqdm

"""""
获取文件夹中的部分文件并输出到另外一个文件夹中（shuffle）
"""""

def shuffle_and_extract_files(src_dir, dest_dir, num_files=500):
    # 获取源文件夹中的所有文件
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # 随机打乱文件列表
    random.shuffle(all_files)

    # 提取前 num_files 个文件
    selected_files = all_files[:num_files]

    # 确保目标文件夹存在
    os.makedirs(dest_dir, exist_ok=True)

    # 复制文件到目标文件夹
    for file_name in tqdm(selected_files):
        src_file = os.path.join(src_dir, file_name)
        dest_file = os.path.join(dest_dir, file_name)
        shutil.copy2(src_file, dest_file)

    print(f"提取了 {num_files} 个文件到 {dest_dir}.")


# 使用示例
src_directory = "/media/hkuit164/Backup/coco/train2017"  # 替换为你的源文件夹路径
dest_directory = "/media/hkuit164/WD20EJRX/Aiden/coco"  # 替换为你的目标文件夹路径
shuffle_and_extract_files(src_directory, dest_directory)
