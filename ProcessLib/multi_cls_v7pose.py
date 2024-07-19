import os
import numpy as np

label_studio_txt_file = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/cls_sources_0710/YOLOv7_output/labels"
v7pose_txt_file = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_sources/pose_sources_0710/YOLOv7ps_trainable/labels"
output_txt_file = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources"

os.makedirs(f"{output_txt_file}/train", exist_ok=True)
os.makedirs(f"{output_txt_file}/val", exist_ok=True)

def box_iou_xywh(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def list_files_in_directory(directory_path):
    file_names = []
    for file in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, file)):
            file_names.append(file)
    return file_names


def read_lb_txt_to_dict(file_path):
    result_lb_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            cls = line.split(" ")[0]
            bbox = [format(float(x), '.6f') for x in line.split(" ")[1:5]]
            if not line:
                continue
            result_lb_dict[tuple(bbox)] = cls
    return result_lb_dict


def read_txt_to_dict(file_path):
    result_dict = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            pose = line.split(" ")[5:]
            bbox = [float(x) for x in line.split(" ")[1:5]]
            if not line:
                continue
            result_dict[tuple(bbox)] = pose
    return result_dict


lb_txts = list_files_in_directory(label_studio_txt_file)
train_1cls_txts = list_files_in_directory(f"{v7pose_txt_file}/train")
val_1cls_txts = list_files_in_directory(f"{v7pose_txt_file}/val")
files_ouput = []
for txts in [train_1cls_txts, val_1cls_txts]:
    for txt in txts:
        for lb_txt in lb_txts:
            if lb_txt.split("-")[-1] == txt.split("-")[-1]:
                files_ouput.append(lb_txt)
                print(f"Found matching lb_txt: {lb_txt} for txt: {txt}")
                lb_txt_content = read_lb_txt_to_dict(f"{label_studio_txt_file}/{lb_txt}")
                try:
                    txt_content = read_txt_to_dict(f"{v7pose_txt_file}/train/{txt}")
                    train_file = True
                except:
                    txt_content = read_txt_to_dict(f"{v7pose_txt_file}/val/{txt}")
                    train_file = False
                if train_file is True:
                    cls_modify = open(f"{output_txt_file}/train/{txt}", "a", encoding="utf-8")
                else:
                    cls_modify = open(f"{output_txt_file}/val/{txt}", "a", encoding="utf-8")
                for txt_line, txt_value in txt_content.items():
                    for lb_txt_line, lb_txt_value in lb_txt_content.items():
                        lines = []
                        lb_txt_line_float = tuple(map(float, lb_txt_line))
                        if txt_line == lb_txt_line_float:
                            lines.append(lb_txt_value)
                            lines.extend(lb_txt_line)
                            lines.extend(txt_value)
                            output_line = " ".join(lines)
                            if train_file is True:
                                cls_modify.write(output_line)
                                continue
                            else:
                                cls_modify.write(output_line)
                                continue
                        else:
                            box_lb_txt_line_float = [float(i) for i in lb_txt_line]
                            box_txt_line = [float(i) for i in txt_line]
                            if box_iou_xywh(box_lb_txt_line_float, box_txt_line) >= 0.8:
                                lines.append(lb_txt_value)
                                lines.extend(lb_txt_line)
                                lines.extend(txt_value)
                                output_line = " ".join(lines)
                                if train_file is True:
                                    cls_modify.write(output_line)
                                else:
                                    cls_modify.write(output_line)
                cls_modify.close()

diff_files = list(set(lb_txts).difference(set(train_1cls_txts+val_1cls_txts))) + list(set(train_1cls_txts+val_1cls_txts).difference(set(lb_txts)))
print("Different with yolov7 and yolov7pose: ", diff_files)
print("Number of different file: ", len(diff_files))

print("Number of file ouput: ", len(files_ouput))
