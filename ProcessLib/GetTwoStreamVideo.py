import cv2
import argparse

# RgbVideo_address = "/media/hkuit164/Backup/rgb.mp4"
# thermalVideo_address = "/media/hkuit164/Backup/thermal.mp4"

def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--folder',
                        help='folder name',
                        required=True,
                        type=str)

    parser.add_argument('--name',
                        help="file name",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()

RgbVideo_address = args.folder + args.name + "_rgb.mp4"
thermalVideo_address = args.folder + args.name + "_thermal.mp4"

if __name__ == '__main__':

    # 開啟 RTSP 串流
    RgbVideoCap = cv2.VideoCapture('rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/101/?transportmode=unicast --input-rtsp-latency=0')
    TherVideoCap = cv2.VideoCapture('rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/201/?transportmode=unicast --input-rtsp-latency=0')

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 15

    save_size = (1080,720)
    RgbVideo = cv2.VideoWriter(RgbVideo_address, fourcc, fps, save_size)
    thermalVideo = cv2.VideoWriter(thermalVideo_address, fourcc, fps, save_size)
    #thermalVideo.set(cv2.CAP_PROP_BUFFERSIZE,0)
    #concatVideo = cv2.VideoWriter("/media/hkuit164/Backup/concatVideo.mp4", fourcc, fps, save_size)
    # 建立視窗
    cv2.namedWindow('Rgb_display', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Thermal_display', cv2.WINDOW_AUTOSIZE)

    for i in range(30):
        ret, Rgbimage = RgbVideoCap.read()

    while True:
        # 從 RTSP 串流讀取一張影像
        ret, Rgbimage = RgbVideoCap.read()
        ret, Therimage = TherVideoCap.read()
        if ret:
            # 顯示影像
            #im_v = cv2.vconcat([Rgbimage, Therimage])
            cv2.imshow('Rgb_display', cv2.resize(Rgbimage, (720, 540)))
            RgbVideo.write(cv2.resize(Rgbimage, save_size))
            #concatVideo.write(cv2.resize(im_v, (1080, 720)))
            cv2.moveWindow('Rgb_display', 0,0)
            cv2.imshow('Thermal_display', cv2.resize(Therimage,  (720, 540)))
            thermalVideo.write(cv2.resize(Therimage, save_size))
            cv2.moveWindow('Thermal_display', 740,0)
            key = cv2.waitKey(1)
        if key == ord('q'):
            # 若沒有影像跳出迴圈
            break

    # 釋放資源
    RgbVideoCap.release()
    TherVideoCap.release()

    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()

