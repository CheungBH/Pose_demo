import os
import shutil
import random


def copy_files_to_new_folder(source_folder, destination_folder):

    os.makedirs(destination_folder, exist_ok=True)

    items = os.listdir(source_folder)
    for item in items:
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path)

        elif os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)


def yolo_format_change(input_folder, class_num, data_folder, split):

    with open(os.path.join(input_folder, 'rgb.data'), 'w') as rgbdata:
        rgbdata.write(f"classes={class_num}\ntrain=./data/{data_folder}/train.txt\nvalid=./data/{data_folder}/val.txt\nnames=./data/{data_folder}/rgb.names")

    os.rename(os.path.join(input_folder, "images"), os.path.join(input_folder, "JPEGImages"))
    os.rename(os.path.join(input_folder, "labels"), os.path.join(input_folder, "txt"))
    os.rename(os.path.join(input_folder, "classes.txt"), os.path.join(input_folder, "rgb.names"))
    os.remove(os.path.join(input_folder, "notes.json"))

    images = [f for f in os.listdir(os.path.join(input_folder, "JPEGImages"))
              if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    if split is True:
        random.shuffle(images)
        train_size = int(0.8 * len(images))
        train_images = images[:train_size]
        val_images = images[train_size:]

        with open(os.path.join(input_folder, 'train.txt'), 'w') as train:
            for train_image in train_images:
                train.write(f"data/{data_folder}/JPEGImages/{train_image}\n")

        with open(os.path.join(input_folder, 'val.txt'), 'w') as val:
            for val_image in val_images:
                val.write(f"data/{data_folder}/JPEGImages/{val_image}\n")

    else:
        with open(os.path.join(input_folder, 'train.txt'), 'w') as train:
            for image in images:
                train.write(f"data/{data_folder}/JPEGImages/{image}\n")
        # to be continued


label_studio_yolo = '/media/hkuit164/Backup/xjl/label_studio_yolo/test'
output_yolo = '/media/hkuit164/Backup/xjl/label_studio_yolo/test_format'
cls_num = 4
data_name = "yolo"
data_split = True

copy_files_to_new_folder(label_studio_yolo, output_yolo)
yolo_format_change(output_yolo, cls_num, data_name, data_split)
