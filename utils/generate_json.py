import json


class JsonGenerator:
    def __init__(self, json_name):
        if not json_name:
            json_name = "result.json"
        self.file = open(json_name)
        self.result = {}

    def update(self, idx, boxes, kps, kps_scores, cnt):
        idx, boxes, kps, kps_scores = idx.tolist(), boxes.tolist(), kps.tolist(), kps_scores.tolist()
        current_res = {}
        for i, box, kp, kp_score in zip(idx, boxes, kps, kps_scores):
            tmp = dict()
            tmp["box"], tmp["kp"], tmp["kp_score"] = box, kp, kp_score
            current_res[idx] = tmp
        self.result[cnt] = current_res

    def release(self):
        self.file.write(json.dumps(self.result))



