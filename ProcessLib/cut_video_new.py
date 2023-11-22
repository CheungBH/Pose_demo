import os
import cv2

input_folder = "/media/hkuit164/Backup/xjl/hh_video_data/cut_video_selected/fight_video"
output_folder = "/media/hkuit164/Backup/xjl/hh_video_data/cut_video_selected/fight_video"
cut_frame = 10

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(input_folder, filename)
        output_subfolder = os.path.join(output_folder, os.path.splitext(filename)[0])
        os.makedirs(output_subfolder, exist_ok=True)

        video = cv2.VideoCapture(video_path)

        frame_count = 0
        image_count = 1

        while True:
            ret, frame = video.read()

            if not ret:
                break

            if frame_count % cut_frame == 0:
                output_image_path = os.path.join(output_subfolder, f"{os.path.splitext(filename)[0]}_{frame_count}.jpg")
                cv2.imwrite(output_image_path, frame)
                image_count += 1

            frame_count += 1

        video.release()

        print(f"Video {filename} processed. Total frames: {frame_count}, Cropped images: {image_count - 1}")
