import os, torch

device = "cpu"  # "cuda:0" or "cpu"

yolo_pose = False
# If choosing yolo_pose, you only need to assign detectors path.

detector_cfg = ""
# Assign detector cfg to empty if you are using yolov7
detector_weight = "/Users/cheungbh/Documents/lab_code/yolov7/weights/yolov7.pt"
detector_label = ""

RgbVideoCap = 'rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/101/?transportmode=unicast --input-rtsp-latency=0'
TherVideoCap = 'rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/201/?transportmode=unicast --input-rtsp-latency=0'

pose_weight = "demo_assets/pose/latest.pth"
pose_model_cfg = ""
pose_data_cfg = ""

sort_type = "sort"# "sort" or "deepsort"
deepsort_weight = "demo_assets/deepsort/ckpt.t7"

classifiers_type = [] # "ML", "image" or "sequence"
classifiers_weight = ["demo_assets/ml/SVM_model.joblib",
                      "demo_assets/ml/AdaBoost_model.joblib",
                      "demo_assets/image_cls/demo1/best_acc.pth",
                      "demo_assets/image_cls/demo1/best_acc.pth"]
classifiers_config = ["",
                      "",
                      "config/image_cls_config/shufflenet_rawcrop.json",
                      "config/image_cls_config/shufflenet_rawWhole.json"]
classifiers_label = ["demo_assets/ml/label",
                     "demo_assets/ml/label",
                     "demo_assets/image_cls/demo1/labels.txt",
                     "demo_assets/image_cls/demo1/labels.txt"]


pose3d_config = ""
pose3d_weight = ""

output_src = "output.mp4"
input_src = "asset/video/video_input3.mp4"
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
