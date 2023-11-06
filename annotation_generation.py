# from frame_process import FrameProcessor
from config import config as config
import os
import cv2
from src.human_detection import HumanDetector
from utils.visualize import Visualizer
import json

image_ext = ["jpg", "jpeg", "webp", "bmp", "png", "JPG"]
video_ext = ["mp4", "mov", "avi", "mkv", "MP4"]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 12

save_size = config.out_size
show_size = config.show_size
show = True


class AnnotationJsonGenerator:
    def __init__(self, json_name):
        if not json_name:
            json_name = "result.json"
        self.file = open(json_name, "w")
        self.save_image = 0
        self.image_count = 0
        self.anno_count = 0

        self.JsonData = {}
        self.JsonData["info"] = {}
        self.JsonData["licenses"] = []
        self.images=[]
        self.JsonData["images"] = self.images
        self.JsonData["annotations"] = []
        self.JsonData["categories"] = []
        Category = {"supercategory": "person",
                    "id": 1,
                    "name": "person",
                    "keypoints": [
                        "nose","left_eye","right_eye","left_ear","right_ear",
                        "left_shoulder","right_shoulder","left_elbow","right_elbow",
                        "left_wrist","right_wrist","left_hip","right_hip",
                        "left_knee","right_knee","left_ankle","right_ankle"
                    ],
                    "skeleton": [
                        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                        [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
                    ]
                }
        self.JsonData["categories"].append(Category)
        self.base_kps = 17


    def update(self, idx, boxes, kps, kps_scores, name, img_h, img_w):

        image = {}
        image["file_name"] = name
        image["id"] = self.image_count
        image["height"] = img_h
        image["width"] = img_w
        self.images.append(image)

        # If no bounding boxes, do not save frame
        # item = {}
        if len(idx) == 0:
            self.save_image = 0
            # return
        else:
            self.save_image = 1
            idx, boxes, kps, kps_scores = idx.tolist(), boxes.tolist(), kps.tolist(), kps_scores.tolist()

            for i, box, kp, kp_score in zip(idx, boxes, kps, kps_scores):
                item = {}
                remaining_kps = self.base_kps - len(kp)
                head_kp = kp[0]
                for _ in range(remaining_kps):
                    kp.insert(1, head_kp)

                bw, bh = box[2] - box[0], box[3] - box[1]
                item["bbox"] = [box[0], box[1], bw, bh]

                keypoints =[]
                kp_num = 0
                for Keypoint in kp:
                    Keypoint[0] = Keypoint[0]
                    Keypoint[1] = Keypoint[1]

                    keypoints.extend(Keypoint)
                    if 0 < kp_num < 5:
                        keypoints.extend([0])
                    else:
                        keypoints.extend([2])
                    kp_num += 1
                item["keypoints"] = keypoints

                item["id"] = self.anno_count
                self.anno_count += 1
                item["area"] = box[2] * box[3]
                item["category_id"] =  1
                item["iscrowd"] = 0
                item["num_keypoints"] = 17
                item["image_id"] = self.image_count
                self.JsonData["annotations"].append(item)

        self.image_count += 1

    def release(self):
        self.file.write(json.dumps(self.JsonData, indent=2))


class Demo:
    def __init__(self, input_src, output_src):
        self.FP = FrameProcessor()
        self.input = input_src
        self.output = output_src
        self.show = show
        self.save_size = save_size
        self.show_size = show_size
        if os.path.isdir(self.input):
            self.demo_type = "image_folder"
            self.input_imgs = [os.path.join(self.input, file_name) for file_name in os.listdir(self.input)]
            if self.output:
                assert os.path.isdir(self.output), "The output should be a folder when the input is a folder!"
                os.makedirs(self.output, exist_ok=True)
                self.output_imgs = [os.path.join(self.output, file_name) for file_name in os.listdir(self.input)]
            else:
                raise ValueError("Unrecognized src: {}".format(self.input))

    def run(self):
        if self.demo_type == "image_folder":
            for idx, img_name in enumerate(self.input_imgs):
                frame = cv2.imread(img_name)
                frame = self.FP.process(frame, img_name)
                if self.show:
                    cv2.imshow("result", cv2.resize(frame, self.show_size))
                    cv2.waitKey(1)
                if self.output:
                    cv2.imwrite(self.output_imgs[idx], cv2.resize(frame, self.save_size))
            self.FP.release()
        else:
            raise ValueError


detector_cfg = "/media/hkuit164/Backup/2324_data/yolo_rgb/yolov3-1cls.cfg"
detector_weight = "/media/hkuit164/Backup/2324_data/yolo_rgb/last.pt"
detector_label = ""

pose_weight = "/media/hkuit164/Backup/PoseTrainingPytorch_1/exp/RGB/bs8_R50/latest.pth"
pose_model_cfg = ""
pose_data_cfg = ""

deepsort_weight = ""
sort_type = "sort"


class FrameProcessor:
    def __init__(self, json_path="/media/hkuit164/Backup/2324_data/0208_high/rgb/result.json"):
        self.HP = HumanDetector(detector_cfg, detector_weight, pose_weight, pose_model_cfg,
                                pose_data_cfg, "sort", "", "", "", "", "", debug=False, device="cpu")
        self.Json = AnnotationJsonGenerator(json_path)
        self.visualizer = Visualizer(self.HP.estimator.kps, detector_label)

    def process(self, frame, name):
        ids, boxes, boxes_cls, kps, kps_scores = self.HP.process(frame, print_time=True)
        self.visualizer.visualize(frame, ids, boxes, boxes_cls, kps, kps_scores)
        self.HP.init_trackers()
        self.Json.update(ids, boxes, kps, kps_scores, name.split("/")[-1], frame.shape[0], frame.shape[1])

    def release(self):
        self.Json.release()


if __name__ == '__main__':
    # import config as config
    input_src = "/media/hkuit164/Backup/2324_data/0208_high/rgb/images"
    output_src = "/media/hkuit164/Backup/2324_data/0208_high/rgb/output"
    demo = Demo(input_src, output_src)
    demo.run()
