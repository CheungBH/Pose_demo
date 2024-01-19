from demo import Demo

import os


if __name__ == '__main__':
    input_folder = ""
    output_folder = ""

    input_videos = [file_name for file_name in os.listdir(input_folder)]
    for input_video in input_videos:
        input_video_path = os.path.join(input_folder, input_video)
        output_video_path = os.path.join(output_folder, input_video)
        demo = Demo(input_video_path, output_video_path)
        demo.run()
