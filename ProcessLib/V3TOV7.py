import os
import shutil

yolov3_path = '/media/hkuit164/Backup/Yolov3Pruning/data/Data_0_yolo'
yolov7_path = '/media/hkuit164/Backup/yolov7/datasets/ball_detection'
data_name = "ball_detection"


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


def v3_to_v7(input_folder, data_folder):
    images = [f for f in os.listdir(os.path.join(input_folder, "JPEGImages"))
              if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    train_img_folder = os.path.join(input_folder, "images", "train")
    val_img_folder = os.path.join(input_folder, "images", "val")

    os.makedirs(train_img_folder)
    os.makedirs(val_img_folder)
    os.rename(os.path.join(input_folder, "train.txt"), os.path.join(input_folder, "train1.txt"))
    os.rename(os.path.join(input_folder, "val.txt"), os.path.join(input_folder, "val1.txt"))

    with open(os.path.join(input_folder, 'train1.txt'), 'r') as train:
        with open(os.path.join(input_folder, 'train.txt'), 'w') as trainyolo:
            train_txt = train.read()
            train_images = train_txt.split()
            for train_image in train_images:
                train_image = train_image.split("/")[-1]
                trainyolo.write(f"datasets/{data_folder}/images/train/{train_image}\n")
                shutil.move(os.path.join(input_folder, "JPEGImages", train_image), train_img_folder)

    with open(os.path.join(input_folder, 'val1.txt'), 'r') as val:
        with open(os.path.join(input_folder, 'val.txt'), 'w') as valyolo:
            val_txt = val.read()
            val_images = val_txt.split()
            for val_image in val_images:
                val_image = val_image.split("/")[-1]
                valyolo.write(f"datasets/{data_folder}/images/val/{val_image}\n")
                shutil.move(os.path.join(input_folder, "JPEGImages", val_image), val_img_folder)

    train_label_folder = os.path.join(input_folder, "labels", "train")
    val_label_folder = os.path.join(input_folder, "labels", "val")
    os.makedirs(train_label_folder)
    os.makedirs(val_label_folder)

    for train_image in train_images:
        train_image = train_image.split("/")[-1]
        train_label = os.path.splitext(train_image)[0] + ".txt"
        shutil.move(os.path.join(input_folder, "txt", train_label), train_label_folder)
    for val_image in val_images:
        val_image = val_image.split("/")[-1]
        val_label = os.path.splitext(val_image)[0] + ".txt"
        shutil.move(os.path.join(input_folder, "txt", val_label), val_label_folder)

    with open(os.path.join(input_folder, "rgb.names"), 'r') as cls:
        text = cls.read()
        classes = text.split()
        class_num = len(classes)

    with open(os.path.join(input_folder, 'data.yaml'), 'w') as coco:
        coco.write(f"\ntrain: ./datasets/{data_folder}/train.txt\nval: ./datasets/{data_folder}/val.txt\nnc: {class_num}\nnames: {classes}")

    os.remove(os.path.join(input_folder, "rgb.names"))
    os.remove(os.path.join(input_folder, "rgb.data"))
    os.remove(os.path.join(input_folder, "train1.txt"))
    os.remove(os.path.join(input_folder, "val1.txt"))
    os.rmdir(os.path.join(input_folder, "txt"))
    os.rmdir(os.path.join(input_folder, "JPEGImages"))


copy_files_to_new_folder(yolov3_path, yolov7_path)
v3_to_v7(yolov7_path, data_name)
