import os
from tqdm import tqdm
import time

folder_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/datasets/Li_Yufangfei/6cls_left_right/project-139-at-2024-07-25-02-12-006b9840/labels"

print("------------------------------start rename-----------------------------")
try:
    if os.path.exists(folder_path):
        time_start = time.time()
        with tqdm(os.listdir(folder_path)) as pbar:
            for image in pbar:
                split_list = image.split('-')
                os.rename(os.path.join(folder_path, image), os.path.join(folder_path, split_list[-1]))
        time_end = time.time()
        print("time cost: ", time_end-time_start)
except:
    raise FileNotFoundError("cannot find the folder path")

print("------------------------------end rename-----------------------------")
