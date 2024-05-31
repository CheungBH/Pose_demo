import cv2
import os
import argparse


def create_video_from_images(image_folder, output_video_path, fps=24):

    images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    images.sort()

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video created successfully: {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create video from images in a folder.')
    parser.add_argument('--image_folder', type=str, help='Path to the folder containing images.')
    parser.add_argument('--output_video_path', type=str, help='Path to save the output video.')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second for the video.')
    args = parser.parse_args()

    create_video_from_images(args.image_folder, args.output_video_path, args.fps)
