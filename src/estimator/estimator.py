

from .models.pose_model import PoseModel
from .dataset.transform import ImageTransform
from .dataset.predict import HeatmapPredictor
from .utils.utils import get_option_path, get_corresponding_cfg
import os
import torch
import numpy as np

posenet = PoseModel()


class PoseEstimator:
    out_h, out_w, in_h, in_w = 64, 64, 256, 256

    def __init__(self, model_path, model_cfg="", data_cfg="", show=True, device="cuda"):
        if not model_cfg or not data_cfg:
            model_cfg, data_cfg, _ = get_corresponding_cfg(model_path, check_exist=["data", "model"])

        self.show = show
        self.device = device
        option_file = get_option_path(model_path)
        self.transform = ImageTransform()
        self.transform.init_with_cfg(data_cfg)

        if os.path.exists(option_file):
            option = torch.load(option_file)
            self.out_h, self.out_w, self.in_h, self.in_w = \
                option.output_height, option.output_width, option.input_height, option.input_width
        else:
            if data_cfg:
                self.transform.init_with_cfg(data_cfg)
                self.out_h, self.out_w, self.in_h, self.in_w = \
                    self.transform.output_height, self.transform.output_width, self.transform.input_height,self.transform.input_width
            else:
                pass

        posenet.build(model_cfg)
        self.model = posenet.model
        self.kps = posenet.kps
        self.model.eval()
        posenet.load(model_path)
        self.HP = HeatmapPredictor(self.out_h, self.out_w, self.in_h, self.in_w)

    def estimate(self, img, boxes, batch=4):
        num_batches = len(boxes)//batch
        left_over = len(boxes) % batch
        inputs, img_metas = self.preprocess(img, boxes)

        if self.device != "cpu":
            inputs = inputs.cuda()

        outputs = []
        for num_batch in range(num_batches):
            outputs.append(self.model(inputs[num_batch*batch:(num_batch+1)*batch]))
        outputs.append(self.model(inputs[-left_over:]))
        hms = torch.cat(outputs).cpu().data
        kps, scores = self.HP.decode_hms(hms, img_metas)
        return kps, scores

    def preprocess(self, img, boxes):
        enlarged_boxes = [self.transform.scale(img, box) for box in boxes]
        img_metas = []
        inputs = []
        for box in enlarged_boxes:
            cropped_img = self.transform.SAMPLE.crop(box, img)
            inp, padded_size = self.transform.process_frame(cropped_img, self.out_h, self.out_w, self.in_h, self.in_w)
            inputs.append(inp.tolist())
            img_metas.append({
                "name": cropped_img,
                "enlarged_box": [box[0], box[1], box[2], box[3]],
                "padded_size": padded_size
            })
        return torch.tensor(inputs), img_metas
