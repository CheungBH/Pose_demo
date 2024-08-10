import joblib
import numpy as np
import json
from collections import defaultdict

def normalize_keypoints(keypoints, bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    normalized_keypoints = []

    for (x_kp, y_kp) in keypoints:
        x_norm = (x_kp - x1) / width
        y_norm = (y_kp - y1) / height
        normalized_keypoints.append([x_norm, y_norm])

    return normalized_keypoints


def flatten(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list

class SeqMLClassifier:
    def __init__(self, weight, config, label):
        self.joblib_model = joblib.load(weight)
        self.label = label
        self.parse_config(config)
        with open(self.label, 'r') as file:
            self.classes = file.readlines()
        self.box_dict, self.keypoint_dict = defaultdict(list), defaultdict(list)

    def parse_config(self, config_file):
        with open(config_file, "r") as load_f:
            load_dict = json.load(load_f)
        self.frame_num = load_dict["frame_num"]

    def __call__(self, ids, boxes, kps, kps_exist, **kwargs):
        actions = []

        for i, box, kp in zip(ids, boxes, kps):
            self.box_dict[i].append(box)
            self.keypoint_dict[i].append(kp)
            # print(self.box_dict)
            if len(self.box_dict[i]) > self.frame_num:
                self.box_dict[i] = self.box_dict[i][-self.frame_num:]
            if len(self.keypoint_dict[i]) > self.frame_num:
                self.keypoint_dict[i] = self.keypoint_dict[i][-self.frame_num:]

            if len(self.box_dict[id]) >= self.frame_num:
                inp_boxes = self.box_dict[id]
                inp_kps = self.keypoint_dict[id]
                norm_kps = []
                for i in range(self.frame_num):
                    norm_kps.append(normalize_keypoints(inp_kps[i], inp_boxes[i]))
                train_kps = np.array([flatten(norm_kps)])
                predict_nums = self.joblib_model.predict(train_kps)

                # print(predict_nums)
                actions.append(predict_nums)
            else:
                actions.append(-1)
        return actions


