# use this code after using ExCocoSelcPic.py
import os
import shutil
import json


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


def norm_bbox(image_h, image_w, bbox):
    norm_box = [
        (bbox[0] + bbox[2] / 2) / image_w,
        (bbox[1] + bbox[3] / 2) / image_h,
        bbox[2] / image_w,
        bbox[3] / image_h
    ]
    return norm_box


def norm_kp(image_h, image_w, kps):
    norm_kps = []
    for i, value in enumerate(kps):
        if i % 3 == 0:
            norm_kps.append(value / image_w)
        elif i % 3 == 1:
            norm_kps.append(value / image_h)
        elif i % 3 == 2:
            norm_kps.append(value)
            if value == 0:
                norm_kps[i-1] = 0
                norm_kps[i-2] = 0
    return norm_kps


def json2txt(input_json, lb_folder):
    json_data = json.loads(input_json)
    for image in json_data["images"]:
        img_name = image['file_name']
        img_id = image['id']
        img_height = image['height']
        img_width = image['width']

        for anno in json_data["annotations"]:
            box = anno['bbox']
            kps = anno['keypoints']
            anno_img_id = anno['image_id']

            if img_id == anno_img_id:
                nm_box = norm_bbox(img_height, img_width, box)
                nm_kps = norm_kp(img_height, img_width, kps)
                nm_box_str = ' '.join(format(vl, '.6f') for vl in nm_box)
                nm_kps_str = ' '.join(format(vl, '.6f') for vl in nm_kps)
                txt_name = os.path.splitext(img_name)[0]
                with open(os.path.join(lb_folder, f'{txt_name}.txt'), 'a') as wj:
                    wj.write(f"0 {nm_box_str} {nm_kps_str} \n")


def pose_to_yolov7pose(input_folder, data_folder):

    train_images = [f for f in os.listdir(os.path.join(input_folder, "train")) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    val_images = [f for f in os.listdir(os.path.join(input_folder, "val")) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    with open(os.path.join(input_folder, 'train.txt'), 'w') as train:
        for train_image in train_images:
            train.write(f"datasets/{data_folder}/images/train/{train_image}\n")

    with open(os.path.join(input_folder, 'val.txt'), 'w') as val:
        for val_image in val_images:
            val.write(f"datasets/{data_folder}/images/val/{val_image}\n")

    img_folder = os.path.join(input_folder, "images")
    os.makedirs(img_folder)
    shutil.move(os.path.join(input_folder, "train"), img_folder)
    shutil.move(os.path.join(input_folder, "val"), img_folder)

    train_label_folder = os.path.join(input_folder, "labels", "train")
    val_label_folder = os.path.join(input_folder, "labels", "val")
    os.makedirs(train_label_folder)
    os.makedirs(val_label_folder)

    with open(os.path.join(input_folder, 'train.json'), 'r') as tj:
        tj_data = tj.read()
    with open(os.path.join(input_folder, 'val.json'), 'r') as vj:
        vj_data = vj.read()

    json2txt(tj_data, train_label_folder)
    json2txt(vj_data, val_label_folder)

    os.remove(os.path.join(input_folder, "train.json"))
    os.remove(os.path.join(input_folder, "val.json"))
    cls = 1
    with open(os.path.join(input_folder, 'pose.yaml'), 'w') as coco:
        coco.write(f"\ntrain: ./datasets/{data_folder}/train.txt\nval: ./datasets/{data_folder}/val.txt\nnc: {cls}\nnames: ['person']")


pose_folder = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_sources/pose_sources_0628/pose_trainable_total"
yolov7pose_folder = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_sources/pose_sources_0628/YOLOv7ps_trainable"
data_name = "YOLOv7ps_trainable"

copy_files_to_new_folder(pose_folder, yolov7pose_folder)
pose_to_yolov7pose(yolov7pose_folder, data_name)
