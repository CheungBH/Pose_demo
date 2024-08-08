import os
import cv2
import joblib
import numpy as np

import config.config as config
from util.parse_json import JsonParser
from util.visualize import Visualizer
from collections import defaultdict


# filter_criterion = config.filter_criterion
frame_num = 5

model_path = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/sources/merge_frame_pose_sources/online_tennis_slice1/output_model/knn_cfg_model.joblib"
joblib_model = joblib.load(model_path)
ML_label = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/sources/merge_frame_pose_sources/online_tennis_slice1/classes"
# with open(ML_label, 'r') as file:
#     ML_classes = file.readlines()


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

class FrameProcessorJson:
    def __init__(self, json_path):
        # if not json_path:
        #     json_path = config.input_src[:-len(config.input_src.split(".")[-1])-1] + ".json"
        if not os.path.exists(json_path):
            raise FileNotFoundError("Your json file isn't exist")
        self.Json = JsonParser(json_path)
        self.visualizer = Visualizer(17, det_label=ML_label, bg_type=config.vis_bg_type,
                                     kps_color_type=config.kps_color_type)
        self.box_dict, self.keypoint_dict = defaultdict(list), defaultdict(list)

    def process(self, frame, cnt=0):
        ids, boxes, kps, kps_scores = self.Json.parse(cnt)
        list_ids = ids.tolist()
        list_boxes = boxes.tolist()
        list_kps = kps.tolist()
        box_cls = []
        # id1_train_data =
        for id, box, kp in zip(list_ids, list_boxes, list_kps):
            self.box_dict[id].append(box)
            self.keypoint_dict[id].append(kp)
            # print(self.box_dict)
            if len(self.box_dict[id]) > frame_num:
                self.box_dict[id] = self.box_dict[id][-frame_num:]
            if len(self.keypoint_dict[id]) > frame_num:
                self.keypoint_dict[id] = self.keypoint_dict[id][-frame_num:]
            # print(len(self.box_dict[id]))
            # print(len(self.keypoint_dict[id]))
            if len(self.box_dict[id]) >= frame_num:
                inp_boxes = self.box_dict[id]
                inp_kps = self.keypoint_dict[id]
                norm_kps = []
                for i in range(frame_num):
                    norm_kps.append(normalize_keypoints(inp_kps[i], inp_boxes[i]))
                train_kps = np.array([flatten(norm_kps)])
                predict_nums = joblib_model.predict(train_kps)

                # print(predict_nums)
                box_cls.append(predict_nums)


        return self.visualizer.visualize(frame, ids, boxes, box_cls, kps, kps_scores)


if __name__ == '__main__':
    # pose_weight = "/home/hkuit164/Downloads/pytorch_model_samples/mob3/pytorch/3_best_acc.pth"
    # det_cfg = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/nanodet-coco.yml"
    # det_weight = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/model_last.pth"

    input_src = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/datasets/used_datasets/online_source/1_raw_video/tennis_slice1.mp4"
    json_path = "test.json"
    out_src = "result.mp4"
    cap = cv2.VideoCapture(input_src)
    out = cv2.VideoWriter(out_src, cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))
    FPJ = FrameProcessorJson(json_path)
    idx = 1
    while True:
        ret, frame = cap.read()
        if ret:
            img = FPJ.process(frame, idx)
            cv2.imshow("result", img)
            cv2.waitKey(1)
            out.write(cv2.resize(img, (1280, 720)))
        else:
            break
        idx += 1
        # print(idx)

