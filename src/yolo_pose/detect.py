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
from .utils.general import non_max_suppression, scale_coords

tensor = torch.tensor
from config.config import out_size


class YoloPose:

    def __init__(self, weights, device="cuda:0", img_size=640, conf_thresh=0.8, nms_thresh=0.5):
        self.device = device
        self.model = attempt_load(weights, map_location=device)
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.kps = 17
        self.raw_img_size = (720, 1080, 3)

    def process(self, img):
        img = letterbox(img, self.img_size, stride=self.stride, auto=False)[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(pred, self.conf_thresh, self.nms_thresh, classes=None, agnostic=False,
                                   kpt_label=True)[0]

        # boxes, kps, kps_score = [], [], []
        # for i, det in enumerate(pred):
        if len(pred):
            # pred_len = len(pred)
            scale_coords(img.shape[2:], pred[:, :4], self.raw_img_size, kpt_label=False)
            scale_coords(img.shape[2:], pred[:, 6:], self.raw_img_size, kpt_label=True, step=3)
            boxes = pred[..., :6]
            # kps_origin
            kps_x, kps_y = pred[..., -51::3], pred[..., -50::3]
            # kps_raw = torch.cat((pred[..., -51::3], pred[..., -50::3]), dim=1)#.reshape(-1, self.kps, 2).tolist()
            kps = kps_x[..., 0].unsqueeze(dim=0)
            for kp in range(self.kps):
                kps = torch.cat((kps, kps_x[..., kp].unsqueeze(dim=0)), dim=0)
                kps = torch.cat((kps, kps_y[..., kp].unsqueeze(dim=0)), dim=0)
            kps = kps[1:, :].T
            # kps = kps.view(-1, self.kps, 2).transpose(1, 2).contiguous().view(-1, self.kps * 2)
            # kps = [k.tolist() for i, k in enumerate(pred[0][..., -51::3]) if i % 3 != 0]
            kps_score = pred[..., -49::3]
            # Add a final 0 to the prediction
            boxes = torch.cat((boxes, torch.zeros_like(boxes[..., :1])), dim=-1)
            return boxes, kps.view(-1, self.kps, 2), kps_score.view(-1, self.kps, 1)
            # return tensor([boxes]), tensor([kps]).reshape(pred_len, -1, 2), tensor([kps_score])
        else:
            return [], [], []


