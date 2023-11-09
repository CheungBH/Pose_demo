import os
import cv2
import config.config as config
from utils.parse_json import JsonParser
from utils.visualize import Visualizer

# filter_criterion = config.filter_criterion


class FrameProcessorJson:
    def __init__(self, json_path):
        # if not json_path:
        #     json_path = config.input_src[:-len(config.input_src.split(".")[-1])-1] + ".json"
        if not os.path.exists(json_path):
            raise FileNotFoundError("Your json file isn't exist")
        self.Json = JsonParser(json_path)
        self.visualizer = Visualizer(17, det_label=config.detector_label, bg_type=config.vis_bg_type,
                                     kps_color_type=config.kps_color_type)

    def process(self, frame, cnt=0):
        ids, boxes, kps, kps_scores = self.Json.parse(cnt)
        return self.visualizer.visualize(frame, ids, boxes, [], kps, kps_scores)


if __name__ == '__main__':
    # pose_weight = "/home/hkuit164/Downloads/pytorch_model_samples/mob3/pytorch/3_best_acc.pth"
    # det_cfg = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/nanodet-coco.yml"
    # det_weight = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/model_last.pth"

    input_src = "demo_assets/cut_xjl_wheelchair.mov"
    json_path = "result.json"
    out_src = "result.avi"
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

