import cv2
import os


# videos_folder = "kps_video"
# video_paths = ["{}/{}".format(videos_folder, video) for video in os.listdir(videos_folder)]

# If using a list of video paths
video_paths = ["/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/ocr_score/20240131_xzy_test_yt_2_score.mp4",
               "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/ocr_score/20240131_xzy_test_yt_3_score.mp4",
               "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/ocr_score/20240131_xzy_test_yt_4_score.mp4"
               ]


output_path = "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/ocr_score.mp4"
video_streams = [cv2.VideoCapture(path) for path in video_paths]
print("Total videos to merge: ", len(video_paths))

# Get video properties from the first video stream
fps = video_streams[0].get(cv2.CAP_PROP_FPS)
frame_width = int(video_streams[0].get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_streams[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the total height required for the output video
total_height = frame_height

# Create output video writer

output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, total_height))


frames = []  # List to store frames from all input videos
for stream in video_streams:
    while True:
        ret, frame = stream.read()

        if not ret:
            break

        frames.append(frame)

print("Total frames to merge: ", len(frames))
for frame in frames:

    # Write the merged frame to the output video
    output_video.write(frame)

    # Display the merged frame (optional)
    cv2.imshow("Merged Video", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video streams and writer
for stream in video_streams:
    stream.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
print("Merged video saved to: ", output_path)
