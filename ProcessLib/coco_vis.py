#-*-coding:utf-8-*-

import os
import cv2
import numpy as np

from pycocotools.coco import COCO

json_file = '/home/hkuit155/Downloads/ThermalTemData/testModel/20221216test1025using1208/meragedresult.json'
dataset_dir = '/home/hkuit155/Downloads/ThermalTemData/testModel/20221216test1025using1208/images/'
# json_file = '/media/hkuit164/Backup/ThermalProject/HRNet-Human-Pose-Estimation/data/coco/annotations/person_keypoints_val2017.json'
# dataset_dir = '/media/hkuit164/Backup/ThermalProject/HRNet-Human-Pose-Estimation/data/coco/images/val2017/'
coco = COCO(json_file)
catIds = coco.getCatIds(catNms=['person'])  # catIds=1 表示人这一类
imgIds = coco.getImgIds(catIds=catIds)  # 图片id，许多值

Skeletons = np.array([[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], [6,7],[6,8],
        [8,10],[7,9],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]])
colorlist = [(255,0,0), (255,0,0), (0,255,0), (0,255,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,255), (0,0,255), (255,255,255), (255,255,255),
             (255,255,0), (255,255,0), (255,255,0), (255,255,0), (255,255,0), (255,255,0), (255,255,0)]

show = False
writer = cv2.VideoWriter("output.mp4",cv2.VideoWriter_fourcc(*'XVID'), 20,(1280,720))

for i in range(len(imgIds)):
    # try:
        img = coco.loadImgs(imgIds[i])[0]
        # width = img['height']
        # height = img['width']

        image = cv2.imread(dataset_dir + img['file_name'])
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        annos = coco.loadAnns(annIds)
        coco.showAnns(annos)

        imageShow = image.copy()
        for j in range(len(annos)):
            bbox = annos[j]['bbox']
            x, y, w, h = bbox
            # x = x/height*width
            # y = y/width*height
            # w = w/height*width
            # h = h/width*height
            cv2.rectangle(imageShow, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2)
            for index in range(17):
                startX = annos[j]['keypoints'][(Skeletons[index][0]-1)*3]
                startY = annos[j]['keypoints'][(Skeletons[index][0]-1)*3+1]
                endX = annos[j]['keypoints'][(Skeletons[index][1]-1)*3]
                endY = annos[j]['keypoints'][(Skeletons[index][1]-1)*3+1]
                if startX != 0 and endX !=0 and startY != 0 and endY !=0:
                    cv2.line(imageShow, (int(startX), int(startY)),  (int(endX), int(endY)), colorlist[index], 3)
                #point = (int(annos[i]['keypoints'][index*3]/height*width), int(annos[i]['keypoints'][index*3+1]/width*height))
                point = (int(annos[j]['keypoints'][index*3]), int(annos[j]['keypoints'][index*3+1]))
                cv2.circle(imageShow, point, 1, (0, 0, 255), 5)
            #cv2.line(imageShow, (32,307), (33,100), (255,0,0), 3)
        # 参数为(显示的图片名称，要显示的图片)  必须加上图片名称，不然会报错
        if show:
            cv2.imshow('demo', imageShow)
            cv2.waitKey(0)
        writer.write(imageShow)
writer.release()
    # except:
    #     continue
