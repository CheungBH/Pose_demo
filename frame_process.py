from src.human_detection import HumanDetector
from src.human_detection_multi import HumanDetector as HumanDetector_yolopose
import cv2
from config import config as config
from util.generate_json import JsonGenerator
from util.filter_result import ResultFilterer
from util.visualize import Visualizer


yolo_pose = config.yolo_pose
detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg = config.detector_cfg, \
                                config.detector_weight, config.pose_weight, config.pose_model_cfg, config.pose_data_cfg
detector_label = config.detector_label
write_json = config.write_json
filter_criterion = config.filter_criterion
sort_type, deepsort_weight = config.sort_type, config.deepsort_weight
device = config.device
classifiers_type = config.classifiers_type
classifiers_weight = config.classifiers_weight
classifiers_config = config.classifiers_config
classifiers_labels = config.classifiers_label
bg_type = config.vis_bg_type
kps_color_type = config.kps_color_type
pose3d_config, pose3d_weight = config.pose3d_config, config.pose3d_weight


class FrameProcessor:
    def __init__(self):
        human = HumanDetector if not yolo_pose else HumanDetector_yolopose
        self.HP = human(detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg,
                        sort_type, deepsort_weight, classifiers_type, classifiers_weight, classifiers_config,
                        classifiers_labels, pose3d_config, pose3d_weight, device=device)

        self.write_json = write_json
        if write_json:
            if not config.json_path:
                try:
                    json_path = config.input_src[:-len(config.input_src.split(".")[-1]) - 1] + ".json"
                except:
                    json_path = "result.json"
            else:
                json_path = ""
            self.Json = JsonGenerator(json_path)
        self.filter = ResultFilterer(filter_criterion)
        self.visualizer = Visualizer(17, det_label=detector_label, bg_type=bg_type,
                                     kps_color_type=kps_color_type)

    def process(self, frame, cnt=0):
        ids, boxes, boxes_cls, kps, kps_scores = self.HP.process(frame, print_time=True)
        ids, boxes, boxes_cls, kps, kps_scores = self.filter.filter(ids, boxes, boxes_cls, kps, kps_scores, cnt)
        vis_img = self.visualizer.visualize(frame, ids, boxes, boxes_cls, kps, kps_scores)

        if self.write_json:
            self.Json.update(ids, boxes, kps, kps_scores, cnt)
        return vis_img

    def release(self):
        if self.write_json:
            self.Json.release()


if __name__ == '__main__':
    # pose_weight = "/home/hkuit164/Downloads/pytorch_model_samples/mob3/pytorch/3_best_acc.pth"
    # det_cfg = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/nanodet-coco.yml"
    # det_weight = "/home/hkuit164/Downloads/nanodet_weights/coco/pytorch/model_last.pth"
    img_path = "/media/hkuit164/Elements/data/posetrack18/images/test/000691_mpii_test/000025.jpg"

    FP = FrameProcessor()
    img = cv2.imread(img_path)
    FP.process(img)
    cv2.imshow("result", cv2.resize(img, (1080, 720)))
    cv2.waitKey(0)

