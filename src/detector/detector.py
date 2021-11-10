
from .nanodet.nanodet import NanoDetector
import os
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
        else:
            raise ValueError("{} is not a cfg file!".format(cfg_file))

    def detect(self, frame):
        if self.algo == "nanodet":
            meta, res = self.detector.inference(frame)
            boxes = [box for box in res[0][0] if box[-1] > 0.35]
            return torch.tensor([box + [1, 0] for box in boxes])
