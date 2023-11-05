import argparse
from src.human_detection import HumanDetector
import cv2
import os
from config import config as config

detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg = config.detector_cfg, \
                                config.detector_weight, config.pose_weight, config.pose_model_cfg, config.pose_data_cfg


class FrameProcessor:
    def __init__(self):
        self.HP = HumanDetector(detector_cfg, detector_weight, estimator_weight, estimator_model_cfg, estimator_data_cfg,
                                "sort", "", "", "", "", "", debug=False)

    def process_video_folder(self, video_folder, output_folder, frame_interval):
        for video_file in os.listdir(video_folder):
            video_path = os.path.join(video_folder, video_file)
            if not os.path.isfile(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            video_output_folder = os.path.join(output_folder, os.path.splitext(video_file)[0])
            os.makedirs(video_output_folder, exist_ok=True)
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_interval == 0:
                    ids, boxes, kps, kps_scores = self.HP.process(frame, print_time=True)
                    for i in range(len(ids)):
                        person_id = ids[i]
                        bbox = boxes[i]

                        x_min = int(bbox[0])
                        y_min = int(bbox[1])
                        x_max = int(bbox[2])
                        y_max = int(bbox[3])

                        person_folder = os.path.join(video_output_folder, f"person_{person_id}")
                        os.makedirs(person_folder, exist_ok=True)
                        output_filename = f"frame_{frame_num}_person_{person_id}.jpg"
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
