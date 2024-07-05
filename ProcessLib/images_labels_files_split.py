#Realize the division of multiple files in a folder into specified quantities

import os
import random
import shutil

"""
中文简介：
1. 该程序可以选择按数量分割文件
2. 该程序可以处理训练数据集，例如现在有total的图片和total的labels，可以进行处理使得按比例生成images/train, images/val和labels/train, labels/val
3. 可以选择两个文件夹进行比对，检查labels和images是否对应
"""

# Folder address where the original file is stored
# total txt file path
total_labels_path = "/media/hkuit164/Backup/yolov7/datasets/General_cls_0628/total_labels"
target_floder_path = os.path.dirname(total_labels_path)

# total image file path
total_imamges_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/cls_sources_0628/image_merge_cls_total/total_merge_images"



#The number of files stored in each folder
def file_split_average(total_labels_path, target_floder_path):
    num = 200
    list_ = os.listdir(total_labels_path)
    if num > len(list_):
        print('The length of num must be less than:', len(list_))
        exit()
    if int(len(list_) % num) == 0:
        num_file = int(len(list_) / num)
    else:
        num_file = int(len(list_) / num) + 1
    cnt = 0
    for n in range(1, num_file + 1):  # 创建文件夹
        new_file = os.path.join(target_floder_path + str(n))
        if os.path.exists(new_file + str(cnt)):
            print('The path already exists, please resolve the conflict', new_file)
            exit()
        print('Create folder:', new_file)
        os.mkdir(new_file)
        list_n = list_[num * cnt:num * (cnt + 1)]
        for m in list_n:
            old_path = os.path.join(total_labels_path, m)
            new_path = os.path.join(new_file, m)
            shutil.copy(old_path, new_path)
        cnt += 1
    return

# split 2 file in proportion
def labels_split_proportion(total_labels_path, target_floder_path, train, val):

    # if os.path.exists(target_floder_path):
    #     print('The path already exists, please resolve the conflict: ', target_floder_path)
    #     exit()

    for fileName in train:
        old_path = os.path.join(total_labels_path, fileName)
        train_folder_path = os.path.join(os.path.join(target_floder_path, "labels"), "train")
        if not os.path.exists(train_folder_path):
            os.makedirs(train_folder_path)
        shutil.copy(old_path, train_folder_path)

    for fileName in val:
        old_path = os.path.join(total_labels_path, fileName)
        val_folder_path = os.path.join(os.path.join(target_floder_path, "labels"), "val")
        if not os.path.exists(val_folder_path):
            os.makedirs(val_folder_path)
        shutil.copy(old_path, val_folder_path)
    return

def images_split_proportion(total_imamges_path, target_floder_path, train, val, image_type):
    for fileName in train:
        old_path = os.path.join(total_imamges_path, fileName.split(".")[0] + "." + image_type)
        print("images_old_path: ", old_path)
        train_folder_path = os.path.join(os.path.join(target_floder_path, "images"), "train")
        if not os.path.exists(train_folder_path):
            os.makedirs(train_folder_path)
        shutil.copy(old_path, train_folder_path)

    for fileName in val:
        old_path = os.path.join(total_imamges_path, fileName.split(".")[0] + "." + image_type)
        val_folder_path = os.path.join(os.path.join(target_floder_path, "images"), "val")
        if not os.path.exists(val_folder_path):
            os.makedirs(val_folder_path)
        shutil.copy(old_path, val_folder_path)
    return

def check_image_label():
    # Select the two folders you want to check.
    imageName = os.listdir("/media/hkuit164/Backup/yolov7/datasets/General_cls_0628/images/train")
    labelName = os.listdir("/media/hkuit164/Backup/yolov7/datasets/General_cls_0628/labels/train")
    image_compare = []
    label_compare = []
    for image in imageName:
        image_compare.append(image.split(".")[0])
    for label in labelName:
        label_compare.append(label.split(".")[0])
    print("images: ", image_compare)
    print("labels: ", label_compare)
    print("Different file: ", set(image_compare).difference(set(label_compare)))
    print("Number of different file: ", len(set(image_compare).difference(set(label_compare))))

# choose split method
choice = input("Please select your choice: \n"
               "A: split files\n"
               "B: generate images and labels\n"
               "C: check images and labels\n")

if choice == "A" or choice == "a":
    file_split_average(total_labels_path, target_floder_path)
elif choice == "B" or choice == "b":
    proportion = 0.8
    image_type = "jpg"
    path_dir = os.listdir(total_labels_path)
    train_length = round(len(path_dir) * proportion)
    val_length = round(len(path_dir) * (1 - proportion))
    train = random.sample(path_dir, train_length)
    val = list(set(path_dir).difference(set(train)))
    labels_split_proportion(total_labels_path, target_floder_path, train, val)
    images_split_proportion(total_imamges_path, target_floder_path, train, val, image_type)

elif choice == "C" or choice == "c":
    check_image_label()






