import os, torch

detector_cfg = "/home/hkuit155/Desktop/PortableTennis/assets/yolo/4cls/cfg.cfg"
detector_weight = "/home/hkuit155/Desktop/PortableTennis/assets/yolo/4cls/last.pt"
detector_label = "/home/hkuit155/Desktop/PortableTennis/assets/yolo/4cls/rgb.names"

RgbVideoCap = 'rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/101/?transportmode=unicast --input-rtsp-latency=0'
TherVideoCap = 'rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/201/?transportmode=unicast --input-rtsp-latency=0'

pose_weight = "/home/hkuit155/Desktop/PortableTennis/assets/pose/mob3/mob_bs8_0.001/latest.pth"
pose_model_cfg = ""
pose_data_cfg = ""

#input_src = TherVideoCap
output_src = ""
input_src = "/home/hkuit155/Documents/Highlight2/TOP_100_SHOTS_&_RALLIES_2022_ATP_SEASON_91.mp4"
#output_src = "/media/hkuit164/Backup/free_1_thermal (online-video-cutter.com)_detectionresult.mp4"
out_size = (1280, 720)
show_size = (1280, 720)
show = True

write_json = True
json_path = "result.json"
filter_criterion = {}


'''Inner parameters'''
det_conf = 0.5
det_nms = 0.05

kps_conf = 0.05
option_path = "/".join(pose_weight.split("/")[:-1]) + "/option.pkl"
if os.path.exists(option_path):
    info = torch.load(option_path)
    if "thresh" in info:
        kps_conf = info.thresh
ball_conf = 0.05

max_age = 1
min_hints = 0
iou_thresh = 0.3