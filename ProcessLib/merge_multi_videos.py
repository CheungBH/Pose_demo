import math
import cv2
import numpy as np
import sys


class VideoMerger:
    def __init__(self, output_path, input_paths, notes=[]):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.caps = [cv2.VideoCapture(input_path) for input_path in input_paths]
        self.sizes = (int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.get_output_hw(len(input_paths))
        self.out_height = int(self.sizes[1] * self.row)
        self.out_width = int(self.sizes[0] * self.col)
        self.out = cv2.VideoWriter(output_path, self.fourcc, 20.0, (self.out_width, self.out_height))
        if notes:
            assert len(notes) == len(input_paths), \
                "The length of nick_names should be the same as the length of input_paths!"
            self.nick_names = notes
        else:
            self.nick_names = None

    def get_output_hw(self, number):
        self.row = int(math.sqrt(number))
        self.col = math.ceil(number / self.row)
        self.padding = int(self.row * self.col - number)

    def merge(self, frames_list):
        black_img = np.zeros((self.sizes[1], self.sizes[0], 3), dtype=np.uint8)
        for i in range(self.padding):
            frames_list.append(black_img)
        row_images = []
        for i in range(0, len(frames_list), self.col):
            row_img = np.concatenate(frames_list[i:i+self.col], axis=1)
            row_images.append(row_img)
        out_img = np.concatenate(row_images, axis=0)
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
    input_paths = ["../demo_assets/cut_xjl_wheelchair.mov", "../demo_assets/cut_xjl_fall.mp4",
                   "../demo_assets/reverse_1_thermal.mp4", "../demo_assets/cut_xjl_wheelchair.mov", "../demo_assets/cut_xjl_fall.mp4", ]
    output_path = "merged.mp4"
    notes = ["1", "2", "3", "4", "5"] # [] for not adding notes
    VM = VideoMerger(output_path, input_paths, notes)
    VM.update()


