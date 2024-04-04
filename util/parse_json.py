import json
import torch

tensor = torch.Tensor


class JsonParser:
    def __init__(self, json_path):
        self.file = json.load(open(json_path))

    def parse(self, cnt):
        frame_item = self.file[str(cnt)]
        ids, boxes, kps, kps_scores = [], [], [], []
        for idx, item in frame_item.items():
            ids.append(float(idx))
            boxes.append(item["box"])
            kps.append(item["kp"])
            kps_scores.append(item["kp_score"])
        return tensor(ids), tensor(boxes), tensor(kps), tensor(kps_scores)


if __name__ == '__main__':
    JP = JsonParser("/home/hkuit164/Desktop/ncnn_pose_demo/dev/video/video4_Trim.json")
