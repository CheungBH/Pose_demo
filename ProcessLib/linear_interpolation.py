# use the OpenCV library to apply interpolation algorithms to address video frame skipping issues

import cv2
import numpy as np


def interpolate_frames(prev_frame, next_frame, prev_gray, next_gray, num_frames):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 5, 5, 1.2, 0)
    interpolated_frames = []

    for i in range(1, num_frames + 1):
        alpha = i / (num_frames + 1)
        frame = cv2.addWeighted(next_frame, alpha, prev_frame, 1 - alpha, 0)
        flow_interpolated = -alpha * flow
        h, w = flow_interpolated.shape[:2]
        flow_interpolated[:,:,0] += np.arange(w)
        flow_interpolated[:,:,1] += np.arange(h)[:,np.newaxis]

        frame_interpolated = cv2.remap(frame, flow_interpolated, None, cv2.INTER_LINEAR)
        interpolated_frames.append(frame_interpolated)

    return interpolated_frames

input_video = "/media/hkuit164/Backup/xjl/hh_video_data/ffmpeg_video/fight/h6_012_022.mp4"
output_video = "/media/hkuit164/Backup/xjl/hh_video_data/ffmpeg_video/fight/h6_012_022_fix.mp4"

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, next_frame = cap.read()
    if not ret:
        break

    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    interpolated_frames = interpolate_frames(prev_frame, next_frame, prev_gray, next_gray, 2)

    out.write(prev_frame)
    for frame in interpolated_frames:
        out.write(frame)

    prev_frame = next_frame
    prev_gray = next_gray

cap.release()
out.release()
print("Interpolation processing completed")

