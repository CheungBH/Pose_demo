import json
import os

v7pose_txt_folder = "/media/hkuit164/Backup/yolov7ps/yolov7/runs/detect/exp/labels"
output_json = "/media/hkuit164/Backup/yolov7ps/yolov7/runs/detect/exp/labels.json"  # label studio concise input

json_data = {}
json_data["info"] = {}
json_data["licenses"] = []
json_data["images"] = []
json_data["annotations"] = []
img_id_num = 0
img_width = 1920
img_height = 1080

for file in os.listdir(v7pose_txt_folder):
    if file.endswith(".txt"):
        img_name = file[:-4] + ".jpg"
        json_data["images"].append({"file_name": img_name, "id": img_id_num, "height": img_height, "width": img_width})

        with open(os.path.join(v7pose_txt_folder, file), "r", encoding="utf-8") as f:
            idx = 0
            contents = [line.strip() for line in f.readlines()]
            for content in contents:
                elements = content.split()
                bbox_value = elements[1:5]
                bbox_float = [float(box) for box in bbox_value]
                bbox_float[2] = bbox_float[2]-bbox_float[0]
                bbox_float[3] = bbox_float[3]-bbox_float[1]
                kps = elements[5:56]
                kps_value = [float(kp) for kp in kps]
                kps_float = [2 if (index + 1) % 3 == 0 else x for index, x in enumerate(kps_value)]
                json_data["annotations"].append({"bbox": bbox_float,
                                                 "keypoints": kps_float,
                                                 "image_id": img_id_num,
                                                 "id": idx,
                                                 "category_id": 1,
                                                 "iscrowd": 0,
                                                 "num_keypoints": 17})
                idx += 1
        img_id_num += 1

categories = []
category = {"supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose","left_eye","right_eye","left_ear","right_ear",
                "left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"
            ],
            "skeleton": [
                [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
            ]
            }
categories.append(category)
json_data["categories"] = categories

with open(output_json, "w", encoding="utf-8") as outfile:
    json.dump(json_data, outfile, ensure_ascii=False, indent=4)
