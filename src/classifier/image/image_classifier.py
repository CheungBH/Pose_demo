from .CNN_models import ModelBuilder
from .utils import read_labels, image_normalize, crop, scale
import torch
from .kps_vis import KeyPointVisualizer
import json


class ImageClassifier:
    def __init__(self, weight, config, label, transform, device="cuda:0", max_batch=4):
        self.transform = transform
        self.label = label
        self.model_size = 224
        self.parse_config(config)
        self.classes = read_labels(label)
        self.MB = ModelBuilder()
        self.model = self.MB.build(len(self.classes), self.backbone, device)
        self.MB.load_weight(weight)
        self.model.eval()
        self.max_batch = max_batch
        self.KPV = KeyPointVisualizer(transform.kps, "coco")

    def parse_config(self, config_file):
        with open(config_file, "r") as load_f:
            load_dict = json.load(load_f)
        self.backbone = load_dict["backbone"]
        self.img_type = load_dict["image_type"]
        assert self.img_type in ["black_kps", "raw_crop"]

    def __call__(self, img, boxes, kps, kps_exist):
        img_tns = self.preprocess(img, boxes, kps, kps_exist)
        self.scores = self.MB.inference_tensor(img_tns)
        _, self.pred_idx = torch.max(self.scores, 1)
        self.pred_cls = [self.classes[i] for i in self.pred_idx]
        return self.pred_cls

    def preprocess(self, img, boxes, kps, kps_score):
        if "crop" in self.img_type:
            img = img if self.img_type == "raw_crop" else self.KPV.visualize(img, kps, kps_score)
            imgs_tensor = None
            for box in boxes:
                scaled_box = self.transform.scale(img, box)
                cropped_img = self.transform.SAMPLE.crop(scaled_box, img)
                img_tensor = image_normalize(cropped_img, size=self.model_size)
                imgs_tensor = torch.unsqueeze(img_tensor, dim=0) if imgs_tensor is None else torch.cat(
                    (imgs_tensor, torch.unsqueeze(img_tensor, dim=0)), dim=0)
        elif "whole" in self.img_type:
            if "raw" in self.img_type:
                target_img = img
            else:
                target_img = self.KPV.visualize(img, kps, kps_score)
            img_tensor = image_normalize(target_img, size=self.model_size)
            imgs_tensor = torch.unsqueeze(img_tensor, dim=0)
        else:
            raise ValueError("Unknown image type: {}".format(self.img_type))
        return imgs_tensor



