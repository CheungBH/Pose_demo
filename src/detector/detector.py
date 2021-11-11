from .yolo.yolo import YoloDetector
from .nanodet.nanodet import NanoDetector
from .nanodet.util import Logger, cfg, load_config

import torch


class PersonDetector:
    def __init__(self, cfg_file, weight):
        if cfg_file.endswith(".yml"):
            logger = Logger(0, use_tensorboard=False)
            self.algo = "nanodet"
            load_config(cfg, cfg_file)
            self.detector = NanoDetector(cfg, weight, logger)
        elif cfg_file.endswith(".cfg"):
            self.algo = "yolov3"
            self.detector = YoloDetector(cfg_file, weight)
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


if __name__ == '__main__':
    import cv2
    cfg = "/home/hkuit164/Downloads/yolo_selected/coco_basic/pytorch/yolov3-original-1cls-leaky.cfg"
    weight = "/home/hkuit164/Downloads/yolo_selected/coco_basic/pytorch/last.weights"

    detector = PersonDetector(cfg, weight)

    img_path = "/media/hkuit164/Elements/data/posetrack18/images/test/000693_mpii_test/000013.jpg"
    img = cv2.imread(img_path)
    dets = detector.detect(img)
    for bbox in dets:
        bbox = bbox[:4].tolist()
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
    cv2.imshow("result", img)
    cv2.waitKey(0)
