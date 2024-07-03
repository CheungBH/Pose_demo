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


def yolo_format_change(input_folder, data_folder, split):
    with open(os.path.join(input_folder, "classes.txt"), 'r') as cls:
        text = cls.read()
        classes = text.split()
        class_num = len(classes)

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


def yolov7_format_change(input_folder, split, data_folder):
    os.remove(os.path.join(input_folder, "notes.json"))

    images = [f for f in os.listdir(os.path.join(input_folder, "images"))
              if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    if split is True:
        random.shuffle(images)
        train_size = int(0.8 * len(images))
        train_images = images[:train_size]
        val_images = images[train_size:]

        with open(os.path.join(input_folder, 'train.txt'), 'w') as train:
            for train_image in train_images:
                train.write(f"datasets/{data_folder}/images/train/{train_image}\n")

        with open(os.path.join(input_folder, 'val.txt'), 'w') as val:
            for val_image in val_images:
                val.write(f"datasets/{data_folder}/images/val/{val_image}\n")

        train_img_folder = os.path.join(input_folder, "images", "train")
        val_img_folder = os.path.join(input_folder, "images", "val")

        os.makedirs(train_img_folder)
        os.makedirs(val_img_folder)

        for train_image in train_images:
            shutil.move(os.path.join(input_folder, "images", train_image), train_img_folder)
        for val_image in val_images:
            shutil.move(os.path.join(input_folder, "images", val_image), val_img_folder)

        train_label_folder = os.path.join(input_folder, "labels", "train")
        val_label_folder = os.path.join(input_folder, "labels", "val")
        os.makedirs(train_label_folder)
        os.makedirs(val_label_folder)

        for train_image in train_images:
            train_label = os.path.splitext(train_image)[0] + ".txt"
            shutil.move(os.path.join(input_folder, "labels", train_label), train_label_folder)
        for val_image in val_images:
            val_label = os.path.splitext(val_image)[0] + ".txt"
            shutil.move(os.path.join(input_folder, "labels", val_label), val_label_folder)

    with open(os.path.join(input_folder, "classes.txt"), 'r') as cls:
        text = cls.read()
        classes = text.split()
        class_num = len(classes)

    with open(os.path.join(input_folder, 'data.yaml'), 'w') as coco:
        coco.write(f"\ntrain: ./datasets/{data_folder}/train.txt\nval: ./datasets/{data_folder}/val.txt\nnc: {class_num}\nnames: {classes}")

    os.remove(os.path.join(input_folder, "classes.txt"))


label_studio_yolo = '/media/hkuit164/WD20EJRX/Aiden/hksi/HKSI/YOLOv7_process/YOLO_ouput'
output_yolo = '/media/hkuit164/WD20EJRX/Aiden/hksi/HKSI/YOLOv7_process/YOLO_trainable'
data_name = "YOLO_trainable"
data_split = True
yolov7 = True

copy_files_to_new_folder(label_studio_yolo, output_yolo)
if yolov7 is False:
    yolo_format_change(output_yolo, data_name, data_split)
else:
    yolov7_format_change(output_yolo, data_split, data_name)
