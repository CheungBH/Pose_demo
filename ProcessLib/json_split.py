#Realize the division of multiple files in a folder into specified quantities

import os
import shutil

"""
There are multiple files in a folder. Divide all the files into num parts and create a new folder to put them in
"""

#Folder address where the original file is stored
file_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/cls_sources_0628/relabel_cls_json"
#Folder address where new files are stored
new_file_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/cls_sources_0628/relabel_json_split"


#The number of files stored in each folder
num = 200
list_ = os.listdir(file_path)
if num > len(list_):
    print('The length of num must be less than:', len(list_))
    exit()
if int(len(list_) % num) == 0:
    num_file = int(len(list_) / num)
else:
    num_file = int(len(list_) / num) + 1
cnt = 0
for n in range(1, num_file + 1):  # 创建文件夹
    new_file = os.path.join(new_file_path + str(n))
    if os.path.exists(new_file + str(cnt)):
        print('The path already exists, please resolve the conflict', new_file)
        exit()
    print('Create folder:', new_file)
    os.mkdir(new_file)
    list_n = list_[num * cnt:num * (cnt + 1)]
    for m in list_n:
        old_path = os.path.join(file_path, m)
        new_path = os.path.join(new_file, m)
        shutil.copy(old_path, new_path)
    cnt += 1
