import os
import csv


folder_path = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources/pose_4cls_lr_0726/pose_4cls_merge/labels'
output_csv_folder = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_cls_sources/pose_4cls_lr_0726/pose_4cls_csv"
os.makedirs(output_csv_folder, exist_ok=True)
train_folder = f"{folder_path}/train"
val_folder = f"{folder_path}/val"


def write_csv(folder_path_txt, csv_folder):
    if folder_path_txt.split("/")[-1] == "train":
        csv_path = f"{csv_folder}/train.csv"
    else:
        csv_path = f"{csv_folder}/val.csv"

    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        for file_name in os.listdir(folder_path_txt):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path_txt, file_name)
                with open(file_path, 'r', encoding='utf-8') as txtfile:
                    for line in txtfile:
                        line_content = line.split(" ")
                        class_num = int(line_content[0])
                        boxes = [float(kp) for kp in line_content[1:5]]
                        kps = [float(kp) for kp in line_content[5:-1]]
                        normed_kps = []
                        box_tlw, box_tlh, box_w, box_h = boxes[0]-0.5*boxes[2], boxes[1]-0.5*boxes[3], boxes[2], boxes[3]
                        for idx, kp in enumerate(kps):
                            if idx % 3 == 0:
                                normed_kps.append(str((kp-box_tlw)/box_w) if kp != 0 else 0)
                            elif idx % 3 == 1:
                                normed_kps.append(str((kp-box_tlh)/box_h) if kp != 0 else 0)
                            else:
                                pass
                        write_line = normed_kps + [class_num, os.path.basename(file_path)]
                        csv_writer.writerow(write_line)

write_csv(train_folder, output_csv_folder)
write_csv(val_folder, output_csv_folder)
