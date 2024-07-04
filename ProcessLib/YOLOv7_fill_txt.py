import os
from tqdm import tqdm
import time

folder_path = "/media/hkuit164/Backup/yolov7/datasets/General_cls_0628/images/val"
folder_root = os.path.dirname(os.path.dirname(folder_path))
target_file = os.path.join(folder_root, folder_path.split("/")[-1]) + ".txt"

if os.path.exists(folder_path):
    time_start = time.time()
    with tqdm(os.listdir(folder_path)) as pbar:
        file = open(target_file, mode='w', encoding='utf-8')
        for image in pbar:
            file.write(str(os.path.join(folder_path, image)))
            file.write("\n")
        file.close()
