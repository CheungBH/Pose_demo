
import torch
import numpy as np
import cv2
from .models.experimental import attempt_load
from .utils.datasets import letterbox
from .utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from .utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Yolov7Detector:
    def __init__(self, cfg="", weight="", device="cpu", img_size=640, conf_thresh=0.25, nms_thresh=0.45):
        set_logging()
        self.device = select_device(device)
        self.model = attempt_load(weight, map_location=device)  # load FP32 model
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.img_size = img_size
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.model = TracedModel(self.model, device, img_size)
        self.conf = conf_thresh
        self.nms = nms_thresh

    def process(self, img):
        raw_img_size = (img.shape[0], img.shape[1], img.shape[2])
        img = letterbox(img, self.img_size, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf, self.nms)[0]
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], raw_img_size).round()
        pred = torch.cat((pred[..., :5], torch.ones_like(pred[..., :1]), pred[..., 5:]), dim=-1)
        return pred



