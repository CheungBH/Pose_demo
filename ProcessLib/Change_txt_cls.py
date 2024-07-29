import os
from tqdm import tqdm

"""
    用于修改cls的txt文件中的cls的种类：
"""


txt_folder_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/6cls_lr_0725/YOLO_output/labels"
changed_txt_folder_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/6cls_lr_0725/YOLO_output/new_labels"

def change_cls(cls):
    if cls == "0" or cls == "3":
        return "0"
    elif cls == "1" or cls == "4":
        return "1"
    elif cls == "5":
        return "3"
    else:
        return cls

for txt in tqdm(os.listdir(txt_folder_path)):
    f = open(os.path.join(txt_folder_path, txt), 'r')
    c_f = open(os.path.join(changed_txt_folder_path, txt), 'a+')
    lines = f.readlines()
    for line in lines:
        txt_line = line.strip().split(" ")
        txt_line[0] = change_cls(txt_line[0])
        save_line = ' '.join(txt_line)
        if line == lines[-1]:
            c_f.write(save_line)
        else:
            c_f.write(save_line)
            c_f.write("\n")
    f.close()
    c_f.close()
    print(lines)
