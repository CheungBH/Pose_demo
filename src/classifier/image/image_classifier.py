from .CNN_models import ModelBuilder
from .utils import read_labels, image_normalize, crop, scale
import torch
from .kps_vis import KeyPointVisualizer


class ImageClassifier:
    def __init__(self, weight, config, label, transform, device="cuda:0", max_batch=4, img_type="black_kps"):
        self.label = label
        self.model_size = 224
        self.classes = read_labels(label)
        self.MB = ModelBuilder()
        self.model = self.MB.build(len(self.classes), config, device)
        self.MB.load_weight(weight)
        self.model.eval()
        self.max_batch = max_batch
        self.img_type = img_type
        assert img_type in ["black_kps", "origin_crop"]
        self.sf = [transform.scale_factor/2 for _ in range(4)]
        self.KPV = KeyPointVisualizer(transform.kps, "coco")

    def __call__(self, img, boxes, kps, kps_score):
        img_tns = self.preprocess(img, boxes, kps, kps_score)
        self.scores = self.MB.inference(img_tns)
        _, self.pred_idx = torch.max(self.scores, 1)
        self.pred_cls = self.classes[self.pred_idx]
        return self.pred_cls

    def preprocess(self, img, boxes, kps, kps_score):
        img = img if self.img_type == "origin_crop" else self.KPV.visualize(img, kps, kps_score)
        imgs_tensor = []
        for box in boxes:
            scaled_box = scale(img, box, self.sf)
            cropped_img = crop(scaled_box, img)
            img_tensor = image_normalize(cropped_img, size=self.model_size)
            imgs_tensor.append(img_tensor)
        return imgs_tensor



