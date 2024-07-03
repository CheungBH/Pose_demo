import os, torch

device = "cpu"  # "cuda:0" or "cpu"

yolo_pose = False

# If you are using yolov7, you need to assign yolov7_pose_path OR yolov7_path.
# Otherwise, you will encounter an error "no module named 'models'"

yolov7_pose_path = "/media/hkuit164/Backup/yolov7ps/yolov7"
yolov7_path = "/media/hkuit164/Backup/yolov7"
# If choosing yolo_pose, you only need to assign detectors path.

detector_cfg = ""
# Assign detector cfg to empty if you are using yolov7
detector_weight = "/media/hkuit164/Backup/yolov7/runs/train/yolov7_detection_tennis_1cls3/weights/last.pt"
detector_label = ""

RgbVideoCap = 'rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/101/?transportmode=unicast --input-rtsp-latency=0'
TherVideoCap = 'rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/201/?transportmode=unicast --input-rtsp-latency=0'

pose_weight = "/media/hkuit164/Backup/PortableTennis/assets/pose/mob3/mob_bs4_0.001/latest.pth"
pose_model_cfg = ""
pose_data_cfg = ""

sort_type = "sort"# "sort" or "deepsort"
deepsort_weight = ""

classifiers_type = [] # "ML", "image" or "sequence"
classifiers_weight = []
classifiers_config = []
classifiers_label = []


pose3d_config = ""
pose3d_weight = ""

output_src = "output.mp4"
input_src = "tennis_assets/videos/20240131_xzy_test_yt_10.mp4"
#output_src = "/media/hkuit164/Backup/free_1_thermal (online-video-cutter.com)_detectionresult.mp4"
out_size = (1280, 720)
show_size = (1280, 720)
show = True
vis_bg_type = "raw" # "raw" or "black"
kps_color_type = "COCO" # "per_id" or "COCO"

write_json = True
json_path = "result.json"
filter_criterion = {}


'''Inner parameters'''
det_conf = 0.1
det_nms = 0.05

kps_conf = 0.05
option_path = "/".join(pose_weight.split("/")[:-1]) + "/option.pkl"
if os.path.exists(option_path):
    info = torch.load(option_path)
    if "thresh" in info:
        kps_conf = info.thresh
ball_conf = 0.05

max_age = 5
min_hints = 3
iou_thresh = 0.3
