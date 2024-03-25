from .yolo.yolo import YoloDetector
from .yolov7.detect import Yolov7Detector
# from .nanodet.nanodet import NanoDetector
# from .nanodet.util import Logger, cfg, load_config

import torch


class PersonDetector:
    def __init__(self, cfg_file, weight, device="cuda:0"):
        self.device = device
        if cfg_file.endswith(".yml"):
            logger = Logger(0, use_tensorboard=False)
            self.algo = "nanodet"
            load_config(cfg, cfg_file)
            self.detector = NanoDetector(cfg, weight, logger)
        elif cfg_file.endswith(".cfg"):
            self.algo = "yolov3"
            self.detector = YoloDetector(cfg_file, weight, device=device)
        elif not cfg_file:
            self.algo = "yolov7"
            self.detector = Yolov7Detector(cfg_file, weight, device=device)
        else:
            raise ValueError("{} is not a cfg file!".format(cfg_file))

    def detect(self, frame):
        if self.algo == "nanodet":
            meta, res = self.detector.inference(frame)
            boxes = [box for box in res[0][0] if box[-1] > 0.35]
            return torch.tensor([box + [1, 0] for box in boxes])
        elif self.algo == "yolov3":
            boxes = self.detector.inference(frame)
            return boxes
        elif self.algo == "yolov7":
            boxes = self.detector.inference(frame)
            return boxes


if __name__ == '__main__':
    import cv2
    cfg = ""
    weight = "../../asset/yolo/yolov7.pt"
    img = "/Users/cheungbh/Documents/lab_code/yolov7/inference/images/horses.jpg"

    detector = PersonDetector(cfg, weight, conf_thresh=0.5, nms_thresh=0.5)
    boxes = detector.detect(cv2.imread(img))