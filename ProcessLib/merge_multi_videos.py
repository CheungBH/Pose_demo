import math
import cv2
import numpy as np
import sys


class VideoMerger:
    def __init__(self, output_path, input_paths, notes=[], gap=50, resize_height=None):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.caps = [cv2.VideoCapture(input_path) for input_path in input_paths]
        self.sizes = (int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.get_output_hw(len(input_paths), gap)
        self.out_height = int(self.sizes[1] * self.row + (self.row - 1) * gap)  # Add gap between videos vertically
        self.out_width = int(self.sizes[0] * self.col + (self.col - 1) * gap)  # Add gap between videos horizontally
        self.out = cv2.VideoWriter(output_path, self.fourcc, 20.0, (self.out_width, self.out_height))
        if notes:
            assert len(notes) == len(input_paths), \
                "The length of nick_names should be the same as the length of input_paths!"
            self.nick_names = notes
        else:
            self.nick_names = None
        self.gap = gap
        self.resize_height = resize_height

    def get_output_hw(self, number, gap):
        self.row = int(math.sqrt(number))
        self.col = math.ceil(number / self.row)
        self.padding = int(self.row * self.col - number)

    def merge(self, frames_list):
        black_img = np.zeros((self.sizes[1], self.sizes[0], 3), dtype=np.uint8)
        for i in range(self.padding):
            frames_list.append(black_img)
        row_images = []
        for i in range(0, len(frames_list), self.col):
            row_img = []
            for j in range(self.col):
                frame = frames_list[i + j]
                if self.resize_height:
                    frame = cv2.resize(frame, (int(self.sizes[0] * self.resize_height / self.sizes[1]), self.resize_height))
                row_img.append(frame)
                if j < self.col - 1:
                    row_img.append(np.zeros((self.resize_height if self.resize_height else self.sizes[1], self.gap, 3), dtype=np.uint8) + 255)  # Add white gap horizontally
            row_img = np.concatenate(row_img, axis=1)
            row_images.append(row_img)

        if len(row_images) < self.row:
            for _ in range(self.row - len(row_images)):
                row_images.append(np.zeros((self.resize_height if self.resize_height else self.sizes[1], self.out_width, 3), dtype=np.uint8) + 255)  # 使用白色填充

        out_img = []
        for k in range(len(row_images)):
            out_img.append(row_images[k])
            if k < len(row_images) - 1:
                out_img.append(np.zeros((self.gap, self.out_width, 3), dtype=np.uint8) + 255)  # Add white gap vertically
        out_img = np.concatenate(out_img, axis=0)
        return out_img

    def update(self):
        idx = 0
        while True:
            if idx % 50 == 0:
                print("Processing frame: {}".format(idx))
            idx += 1
            frames = []
            for cap in self.caps:
                ret, frame = cap.read()
                if not ret:
                    self.out.release()
                    print('Some video is over. Finish merging.')
                    sys.exit(0)
                frames.append(frame)
            for frame, nick_name in zip(frames, self.nick_names):
                cv2.putText(frame, nick_name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            merged_frame = self.merge(frames)
            cv2.imshow("merged", merged_frame)
            self.out.write(merged_frame)


if __name__ == '__main__':
    input_paths = ["/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/ball_recognition_20.mp4",
                   "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/ball_landing_20.mp4",
                   "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/highlight_recognition_20.mp4",
                   "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced/ocr_score_20.mp4"]
    output_path = "/media/hkuit164/WD20EJRX/ESTRNN_dataset/Tennis_demo/AI_model_advanced.mp4"
    notes = [" ", " ", " ", " "] # [] for not adding notes
    VM = VideoMerger(output_path, input_paths, notes, gap=200, resize_height=720)
    VM.update()
