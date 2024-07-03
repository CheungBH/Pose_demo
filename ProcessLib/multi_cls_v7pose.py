import os

label_studio_txt_file = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/cls_sources_0628/YOLOv7_trainable"
v7pose_txt_file = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_sources/pose_sources_0628/YOLOv7ps_trainable/labels"
output_txt_file = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources"

os.makedirs(f"{output_txt_file}/train", exist_ok=True)
os.makedirs(f"{output_txt_file}/val", exist_ok=True)


def list_files_in_directory(directory_path):
    file_names = []()
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

for txts in [train_1cls_txts, val_1cls_txts]:
    for txt in txts:
        for lb_txt in lb_txts:
            if lb_txt.split("-")[-1] == txt:
                print(f"Found matching lb_txt: {lb_txt} for txt: {txt}")
                lb_txt_content = read_lb_txt_to_dict(f"{label_studio_txt_file}/{lb_txt}")
                try:
                    txt_content = read_txt_to_dict(f"{v7pose_txt_file}/train/{txt}")
                    train_file = True
                except:
                    txt_content = read_txt_to_dict(f"{v7pose_txt_file}/val/{txt}")
                    train_file = False
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
                                with open(f"{output_txt_file}/train/{txt}", "a", encoding="utf-8") as cls_modify:
                                    cls_modify.write(output_line)
                            else:
                                with open(f"{output_txt_file}/val/{txt}", "a", encoding="utf-8") as cls_modify:
                                    cls_modify.write(output_line)
