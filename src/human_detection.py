import cv2
import torch

from .detector.detector import PersonDetector
from .estimator.estimator import PoseEstimator
from .tracker.tracker import PersonTracker


class HumanDetector:
    def __init__(self, detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg):
        self.detector = PersonDetector(detector_cfg, detector_weight)
        self.estimator = PoseEstimator(estimator_weight, estimator_model_cfg, estimator_data_cfg)
        self.tracker = PersonTracker()

    def process(self, frame):
        self.ids, self.boxes, self.kps, self.kps_scores = [], [], [], []
        dets = self.detector.detect(frame)
        if len(dets) > 0:
            self.ids, self.boxes = self.tracker.track(dets)
            self.kps, self.kps_scores = self.estimator.estimate(frame, self.boxes)
        return self.ids, self.boxes, self.kps, self.kps_scores

    def visualize(self, img):
        if self.boxes:
            from .tracker.visualize import IDVisualizer
            from .estimator.visualize import KeyPointVisualizer
            id2box = self.tracker.get_id2bbox()
            IDVisualizer().plot_bbox_id(id2box, img, with_bbox=True)
            KeyPointVisualizer(self.estimator.kps, "coco").visualize(img, self.kps, self.kps_scores)


if __name__ == '__main__':
    from detector.nanodet.nanodet import NanoDetector
    from detector.nanodet.util import Logger, cfg, load_config
    from tracker.sort import Sort
    from tracker.visualize import IDVisualizer
    from estimator.estimator import PoseEstimator
    from estimator.visualize import KeyPointVisualizer
    from estimator.utils.utils import get_corresponding_cfg

    model_path = "/home/hkuit164/Downloads/pytorch_model_samples/mob3/pytorch/3_best_acc.pth"

    model_cfg = ""
    data_cfg = ""

    if not model_path or not data_cfg:
        model_cfg, data_cfg, _ = get_corresponding_cfg(model_path, check_exist=["data", "model"])

    cfg_file = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/nanodet-coco.yml"
    load_config(cfg, cfg_file)
    model = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/model_last.pth"

    tracker = Sort()

    video_path = "/home/hkuit164/Downloads/pexels-4.mp4"

    cap = cv2.VideoCapture(video_path)
    IDV = IDVisualizer()
    logger = Logger(0, use_tensorboard=False)
    predictor = NanoDetector(cfg, model, logger)

    while True:
        ret, frame = cap.read()

        meta, res = predictor.inference(frame)

        boxes = [box for box in res[0][0] if box[-1] > 0.35]
        id2box_tmp = tracker.update(torch.tensor([box + [1,0] for box in boxes]))
        id2box = {int(box[4]): box[:4] for box in id2box_tmp}
        PE = PoseEstimator(model_cfg, model_path, data_cfg)
        kps, scores = PE.estimate(frame, boxes)
        KPV = KeyPointVisualizer(PE.kps, "coco")
        IDV.plot_bbox_id(id2box, frame, with_bbox=True)
        frame = KPV.visualize(frame, kps, scores)
        cv2.imshow("result", cv2.resize(frame, (1080, 720)))
        cv2.waitKey(1)


