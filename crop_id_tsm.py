import argparse
from src.human_detection import HumanDetector
import cv2
import os
from config import config as config
import sys

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
insert_path = config.yolov7_path
sys.path.insert(0, insert_path)

class FrameProcessor:
    def __init__(self):
        self.HP = HumanDetector(detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg,
                                classifiers_weight, classifiers_config, classifiers_labels, pose3d_config, pose3d_weight,
                                sort_type, deepsort_weight, device,  debug=False)
        self.bbox_dict = {}

    def process_video_folder(self, video_folder, output_folder, frame_interval):
        for video_file in os.listdir(video_folder):
            video_path = os.path.join(video_folder, video_file)
            if not os.path.isfile(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            video_name = os.path.splitext(video_file)[0]
            video_output_folder = os.path.join(output_folder, video_name)
            os.makedirs(video_output_folder, exist_ok=True)
            frame_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_interval == 0:
                    ids, boxes, det_cls, kps, kps_scores = self.HP.process(frame, print_time=True)
                    ids = ids.tolist()
                    for i in range(len(ids)):
                        person_id = ids[i]
                        bbox = boxes[i]
                        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                        if person_id in self.bbox_dict:
                            prev_x_min, prev_y_min, prev_x_max, prev_y_max = self.bbox_dict[person_id]

                            if (x_min > prev_x_max or x_max < prev_x_min or y_min > prev_y_max or y_max < prev_y_min):
                                self.bbox_dict[person_id] = (x_min, y_min, x_max, y_max)
                        else:
                            self.bbox_dict[person_id] = (x_min, y_min, x_max, y_max)

                        x_min, y_min, x_max, y_max = self.bbox_dict[person_id]

                        person_folder = os.path.join(video_output_folder, f"person_{int(person_id)}")
                        os.makedirs(person_folder, exist_ok=True)
                        output_filename = f"{video_name}_frame_{frame_num}_person_{int(person_id)}.jpg"
                        output_path = os.path.join(person_folder, output_filename)

                        cropped_img = frame[y_min:y_max, x_min:x_max]
                        cv2.imwrite(output_path, cropped_img)
                frame_num += 1
            cap.release()
        print("Video frames cropped and saved successfully.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process video frames and crop person images')

    parser.add_argument('--video_folder', type=str, help='Path to the video folder')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--frame_interval', type=int, default=1, help='Interval between processed frames')

    args = parser.parse_args()

    frame_processor = FrameProcessor()
    video_folder = args.video_folder
    output_folder = args.output_folder
    frame_interval = args.frame_interval
    frame_processor.process_video_folder(video_folder, output_folder, frame_interval)

