from frame_process import FrameProcessor
import config.config as config
import os
import cv2

image_ext = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
video_ext = ["mp4", "mov", "avi", "mkv"]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 12

save_size = config.out_size
show_size = config.show_size
show = config.show


class Demo:
    def __init__(self, input_src, output_src):
        self.FP = FrameProcessor()
        self.input = input_src
        self.output = output_src
        self.show = show
        self.save_size = save_size
        self.show_size = show_size
        if os.path.isdir(self.input):
            self.demo_type = "image_folder"
            self.input_imgs = [os.path.join(self.input, file_name) for file_name in os.listdir(self.input)]
            if self.output:
                self.output_imgs = [os.path.join(self.output, file_name) for file_name in os.listdir(self.output)]
        elif isinstance(self.input, int):
            self.demo_type = "video"
            self.cap = cv2.VideoCapture(self.input)
            if self.output:
                self.out = cv2.VideoWriter(self.output, fourcc, fps, save_size)
        else:
            ext = os.path.basename(self.input)
            if ext in image_ext:
                self.demo_type = "image"
                self.input_img = cv2.imread(self.input)
            elif ext in video_ext:
                self.demo_type = "video"
                self.cap = cv2.VideoCapture(self.input)
                if self.output:
                    self.out = cv2.VideoWriter(self.output, fourcc, fps, save_size)
            else:
                raise ValueError("Unrecognized src: {}".format(self.input))

    def run(self):
        if self.demo_type == "video":
            while True:
                ret, frame = self.cap.read()
                if ret:
                    self.FP.process(frame)
                    if self.show:
                        cv2.imshow("result", cv2.resize(frame, self.show_size))
                        cv2.waitKey(1)
                    if self.output:
                        self.out.write(cv2.resize(frame, self.save_size))
                else:
                    self.cap.release()
                    if self.output:
                        self.out.release()
        elif self.demo_type == "image":
            frame = self.input_img
            self.FP.process(frame)
            if self.show:
                cv2.imshow("result", cv2.resize(frame, self.show_size))
                cv2.waitKey(0)
            if self.output:
                cv2.imwrite(self.output, cv2.resize(frame, self.save_size))
        elif self.demo_type == "image_folder":
            for idx, img_name in enumerate(self.input_imgs):
                frame = cv2.imread(img_name)
                self.FP.process(frame)
                if self.show:
                    cv2.imshow("result", cv2.resize(frame, self.show_size))
                    cv2.waitKey(1000)
                if self.output:
                    cv2.imwrite(self.output_imgs[idx], cv2.resize(frame, self.save_size))
        else:
            raise ValueError


if __name__ == '__main__':
    import config.config as config
    input_src = config.input_src
    output_src = config.output_src
    demo = Demo(input_src, output_src)
    demo.run()

