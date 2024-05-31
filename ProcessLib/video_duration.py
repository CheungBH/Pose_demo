import cv2
from moviepy.editor import VideoFileClip, vfx


def adjust_video_duration(input_path, output_path, target_duration):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_duration = frame_count / fps

    cap.release()

    speed_factor = original_duration / target_duration

    clip = VideoFileClip(input_path)
    new_clip = clip.fx(vfx.speedx, speed_factor)
    new_clip.write_videofile(output_path, codec='libx264', fps=fps)


input_video_path = '/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/highlight_recognition/output.mp4'
output_video_path = '//media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/highlight_recognition_20.mp4'
target_duration = 20

adjust_video_duration(input_video_path, output_video_path, target_duration)
