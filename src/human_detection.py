import cv2
import torch
import time
import copy
import imutils
import numpy as np

from .detector.detector import PersonDetector
from .estimator.estimator import PoseEstimator
from .tracker.tracker import PersonTracker
from .classifier.classifier import EnsembleClassifier

tensor = torch.Tensor

from .tracker.visualize import plot_id_box


class HumanDetector:
    def __init__(self, detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg,
                 sort_type, deepsort_weight, classifiers_type, classifiers_weights, classifiers_config, classifiers_label,
                 device="cuda:0", debug=True):
        self.debug = debug
        self.device = device
        self.use_classifier = True if len(classifiers_type) > 0 else False
        if debug:
            self.tracker_map = cv2.VideoWriter("traker_map.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (3000, 1200))
            self.action_map = cv2.VideoWriter("action_map.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (1500, 1200))
        self.detector = PersonDetector(detector_cfg, detector_weight, device)
        self.estimator = PoseEstimator(estimator_weight, estimator_model_cfg, estimator_data_cfg, device=device)
        self.tracker = PersonTracker(sort_type, device=device, model_path=deepsort_weight)
        self.classifier = EnsembleClassifier(classifiers_type, classifiers_weights, classifiers_config,
                                             classifiers_label, self.estimator.transform, device=device)

    def process(self, frame, print_time=False):
        with torch.no_grad():
            self.ids, self.boxes, self.dets_cls, self.kps, self.kps_scores = [], [], [], [], []
            if print_time:
                curr_time = time.time()
            dets = self.detector.detect(frame)
            if print_time:
                print("Detector uses: {}s".format(round((time.time() - curr_time), 4)))
                curr_time = time.time()
            if len(dets) > 0:
                self.dets_cls = dets[:, -1]
                self.ids, self.boxes = self.tracker.update(dets, copy.deepcopy(frame))

                if print_time:
                    print("Tracker uses: {}s".format(round((time.time() - curr_time), 4)))
                    curr_time = time.time()
                if self.boxes:
                    self.kps, self.kps_scores = self.estimator.estimate(frame, self.boxes)
                    if print_time:
                        print("Pose estimator uses: {}s".format(round((time.time() - curr_time), 4)))
                else:
                    return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([])
                if self.use_classifier:
                    self.actions = self.classifier.update(frame, self.boxes, self.kps, self.kps_scores)
                    for idx, action in enumerate(self.actions[0]):
                        cv2.putText(frame, action, (idx * 50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                if self.debug:
                    self.debug_for_tracking(frame)
                    if self.use_classifier:
                        cls_status_img = self.classifier.visualize(self.actions, self.ids, frame)
                        cv2.imshow("action_map", imutils.resize(cls_status_img, width=1000))
                        self.action_map.write(cv2.resize(cls_status_img, (1500, 1200)))

            self.convert_result_to_tensor()
            return self.ids, self.boxes, self.dets_cls, self.kps, self.kps_scores

    def convert_result_to_tensor(self):
        self.ids = tensor(self.ids)
        self.boxes = tensor(self.boxes)
        self.dets_cls = tensor(self.dets_cls)

    def debug_for_tracking(self, frame):
        iou_map = self.tracker.plot_iou_map(np.ones_like(frame))
        pred_map = copy.deepcopy(frame)
        plot_id_box(self.tracker.get_id2bbox(), pred_map, (0, 255, 0), "up")
        plot_id_box(self.tracker.get_pred(), pred_map, (0, 0, 255), "down")
        tracking_map = np.concatenate([iou_map, pred_map], axis=1)
        self.tracker_map.write(cv2.resize(tracking_map, (1500, 1200)))
        cv2.imshow("tracking_map", imutils.resize(tracking_map, width=1000))


    def visualize(self, img):
        if self.boxes:
            from .tracker.visualize import IDVisualizer
            from .estimator.visualize import KeyPointVisualizer
            id2box = self.tracker.get_id2bbox()
            IDVisualizer().plot_bbox_id(id2box, img, with_bbox=True)
            KeyPointVisualizer(self.estimator.kps, "coco").visualize(img, self.kps, self.kps_scores)

    def init_trackers(self):
        self.tracker = PersonTracker("sort")


if __name__ == '__main__':
    from detector.nanodet.nanodet import NanoDetector
    from detector.nanodet.util import Logger, cfg, load_config
    from src.tracker.box_sort.sort import Sort
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


