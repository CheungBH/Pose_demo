import json
import cv2
import sqlite3
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--DetectionJson',
                        help="Detection's output json",
                        required=True,
                        type=str)
    parser.add_argument('--KeypointJson',
                        help="KeypointJson's output json",
                        required=True,
                        type=str)
    parser.add_argument('--OutputJson',
                        help="OutputJson's path",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()
#DetectionJson = "/media/hkuit164/Backup/pose_thermal/result.json"
#KeypointJson = "/media/hkuit164/Backup/pose_thermal/result1.json"
#OutputJson = "/media/hkuit164/Backup/pose_thermal/ImportJson.json"
DetectionJson = args.DetectionJson
KeypointJson = args.KeypointJson
OutputJson = args.OutputJson

class JsonCreator:
    def __init__(self, DetectionJson, KeypointJson):
        self.DetectionJson = DetectionJson
        self.KeypointJson = KeypointJson

    def OutputJson(self):
        return self.CocoJsonCreator("keypoint")

    def CocoJsonCreator(self, DataType):
        JsonData = {}
        JsonData["info"] = {}
        JsonData["licenses"] = []
        images=[]
        i = 0
        filePath = []
        LastFilePath = []
        for index, DetectionItem in enumerate(self.DetectionJson):
            LastFilePath = filePath
            filePath = DetectionItem["image_path"]
            if LastFilePath != filePath and index != 0:
                image = {}
                imageName = os.path.basename(DetectionItem["image_path"])
                image["file_name"] = imageName
                image["id"] = i
                #imageObj = cv2.imread(DetectionItem["image_path"])
                #image["width"] = imageObj.shape[1]
                #image["height"] = imageObj.shape[0]
                image["width"] = DetectionItem["image_info"]["width"][0]
                image["height"] = DetectionItem["image_info"]["height"][0]
                images.append(image)
                i = i + 1
            if i== 0:
                image = {}
                imageName = os.path.basename(DetectionItem["image_path"])
                image["file_name"] = imageName
                image["id"] = i
                image["width"] = DetectionItem["image_info"]["width"][0]
                image["height"] = DetectionItem["image_info"]["height"][0]
                images.append(image)
                i = i + 1
        JsonData["images"] = images
        JsonData["annotations"] = self.CocoAnnotationCreator(DataType)
        JsonData["categories"] = self.CocoCategoriesCreator(DataType)
        return JsonData

    def CocoAnnotationCreator(self, AnnotationType):
        if (AnnotationType == "keypoint"):
            annotations = []
            i = 0
            filePath = []
            LastFilePath = []
            for index, (DetectionItem, KeypointItem) in enumerate(zip(self.DetectionJson, self.KeypointJson[0]["preds"])):
                LastFilePath = filePath
                filePath = DetectionItem["image_path"]
                if LastFilePath != filePath and index != 0:
                    i = i + 1
                annotation = {}
                annotation["bbox"] = [DetectionItem["bbox"][0], DetectionItem["bbox"][1],
                                      DetectionItem["bbox"][2]-DetectionItem["bbox"][0], DetectionItem["bbox"][3]-DetectionItem["bbox"][1]]
                keypoints = []
                for Keypoint in KeypointItem:
                    keypoints.extend([Keypoint[0]+DetectionItem["bbox"][0], Keypoint[1]+DetectionItem["bbox"][1], 2])
                annotation["keypoints"] = keypoints
                annotation["image_id"] = i
                annotation["id"] = index
                annotation["area"] = DetectionItem["bbox"][2] * DetectionItem["bbox"][3]
                annotation["category_id"] = 1
                annotation["iscrowd"] = 0
                annotation["num_keypoints"] = 17
                annotations.append(annotation)
            return annotations

    def CocoCategoriesCreator(self, AnnotationType):
       if (AnnotationType == "keypoint"):
           Categories = []
           Category = {"supercategory": "person",
                        "id": 1,
                        "name": "person",
                        "keypoints": [
                            "nose","left_eye","right_eye","left_ear","right_ear",
                            "left_shoulder","right_shoulder","left_elbow","right_elbow",
                            "left_wrist","right_wrist","left_hip","right_hip",
                            "left_knee","right_knee","left_ankle","right_ankle"
                        ],
                        "skeleton": [
                            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
                        ]
                    }
           Categories.append(Category)
           return Categories

DetectionJsonObj = open(DetectionJson, "r")
DetectionData = json.load(DetectionJsonObj)

KeypointJsonObj = open(KeypointJson, "r")
KeypintData = json.load(KeypointJsonObj)

Merager = JsonCreator(DetectionData, KeypintData)
MeragerJsonData = Merager.OutputJson()
MeragerJsonDataObj = json.dumps(MeragerJsonData, indent=4)

with open(OutputJson, "w") as outfile:
    outfile.write(MeragerJsonDataObj)
