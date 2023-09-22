import json
import cv2
import sqlite3
import os

ProjectId = 47
projectName = "project_47"
DetectionJson = "/media/hkuit164/Backup/pose_thermal/result.json"
KeypointJson = "/media/hkuit164/Backup/pose_thermal/result1.json"

def savefunction(filename,image_info,data):
    conn = sqlite3.connect('/media/hkuit164/Backup/label_studio_consise/label_studio_consise/db.sqlite3')
    print("Opened database successfully")
    cur = conn.cursor()

    cur.execute(''' INSERT INTO image_image(filename,image_info,data,project_id)
              VALUES(?,?,?,?) ''', (filename, json.dumps(image_info), json.dumps(data),ProjectId,))
    conn.commit()
    print('commit')
    #print(filename, image_info, data)

bbox_json = open(DetectionJson, "r")
bbox_data = json.load(bbox_json)

keypoint_json = open(KeypointJson, "r")
keypoint_datas = json.load(keypoint_json)

show = True

tem_name = []
boxes = []
keypoints = {}
filename = []
image_info = []
imag_shape = []
data = {}
i = 0

for b_data, k_data in zip(bbox_data, keypoint_datas[0]["preds"]):

    if tem_name == []:
        tem_name = b_data['image_path']

    if show:
        imag_brg = cv2.imread(b_data['image_path'])
        image_rgb = cv2.cvtColor(imag_brg, cv2.COLOR_BGR2RGB)
        image = image_rgb.copy()

        for k in k_data:
            cv2.circle(image, (int(k[0]+b_data['bbox'][0]), int(k[1]+b_data['bbox'][1])), radius=3, color=(0, 0, 255), thickness=-2)
        cv2.rectangle(image, (b_data['bbox'][0],b_data['bbox'][1]), (b_data['bbox'][2],b_data['bbox'][3]), color=(0, 0, 255), thickness=2)
        cv2.imshow('demo', image)
        cv2.waitKey(0)

    image_name = b_data['image_path'].replace('/media/hkuit164/Backup/label_studio_consise/label_studio_consise/media/{}/'.format(projectName),'')
    if tem_name != b_data['image_path']:
        data = dict(boxes = boxes, keypoint = keypoints)
        savefunction(filename, image_info, data)
        tem_name = b_data['image_path']
        boxes = []
        keypoints = {}
        i = 0

        filename = os.path.join(projectName,image_name)
        imag_shape = cv2.imread(b_data['image_path']).shape
        image_info =dict(height = imag_shape[0], width = imag_shape[1])
        boxes.append([b_data['bbox'][0]/imag_shape[1], b_data['bbox'][1]/imag_shape[0], (b_data['bbox'][2]-b_data['bbox'][0])/imag_shape[1],
                     (b_data['bbox'][3]-b_data['bbox'][1])/imag_shape[0]])
        keypoint_data = []
        for k in k_data:
            keypoint_data.append([(k[0]+b_data['bbox'][0])/imag_shape[1], (k[1]+b_data['bbox'][1])/imag_shape[0],1])
        keypoints[i] = keypoint_data
        i = i + 1
    else:
        if filename == []:
            filename = os.path.join(projectName,image_name)
            imag_shape = cv2.imread(b_data['image_path']).shape
            image_info = dict(height = imag_shape[0], width = imag_shape[1])

        boxes.append([b_data['bbox'][0]/imag_shape[1], b_data['bbox'][1]/imag_shape[0], (b_data['bbox'][2]-b_data['bbox'][0])/imag_shape[1],\
                     (b_data['bbox'][3]-b_data['bbox'][1])/imag_shape[0]])

        keypoint_data = []
        for k in k_data:
            keypoint_data.append([(k[0]+b_data['bbox'][0])/imag_shape[1], (k[1]+b_data['bbox'][1])/imag_shape[0],1])
        keypoints[i] = keypoint_data
        i = i + 1

data = dict(boxes = boxes, keypoint = keypoints)
savefunction(filename,image_info,data)
