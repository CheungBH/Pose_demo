import argparse
import time
from pathlib import Path

import os
import copy
import cv2
import numpy as np
import torch
from .utils.datasets import letterbox
from .models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

tensor = torch.tensor


class YoloPose:

    def __init__(self, weights, device="cuda:0", img_size=640, conf_thresh=0.8, nms_thresh=0.5):
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.kps = 17

    def process(self, img):
        img = letterbox(img, self.img_size, stride=self.stride, auto=False)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thresh, self.nms_thresh, classes=None, agnostic=False,
                                   kpt_label=True)

        boxes, kps, kps_score = [], [], []
        pred_len = len(pred)
        for i, det in enumerate(pred):
            if len(det):
                scale_coords(img.shape[2:], det[:, :4], img.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], img.shape, kpt_label=True, step=3)
                boxes.append(det[:, :6].tolist + [1])
                kps.append([k.tolist() for i, k in enumerate(det[:, 6:]) if (i + 1) % 3 == 0])
                kps_score.append([k.tolist() for i, k in enumerate(det[:, 6:]) if (i + 1) % 3 != 0])
        return tensor([boxes]), tensor([kps]).reshape(pred_len, -1, 2), tensor([kps_score])


