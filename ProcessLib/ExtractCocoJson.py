import os
import json
import random
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--JsonPath',
                        help="Detection's output json",
                        required=True,
                        type=str)
    parser.add_argument('--ExtractRange',
                        help="extract the data from bottom",
                        nargs='+',
                        required=True,
                        type=int)
    parser.add_argument('--OutputJson',
                        help="OutputJson's path",
                        required=True,
                        type=str)
    parser.add_argument('--Shuffle',
                        help="shuffle the output",
                        default = False,
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()

# JsonPath = "/media/hkuit164/Backup/ThermalProject/2022_11_03_cloudy_dataset/thermal/annotations/result_project_58.json"
# ExtractRange = 15
# OutputJson = "/media/hkuit164/Backup/ThermalProject/2022_11_03_cloudy_dataset/thermal/annotations/val.json"
# Shuffle = True

JsonPath = args.JsonPath
ExtractRange = args.ExtractRange
OutputJson = args.OutputJson
Shuffle = args.Shuffle

JsonObj = open(JsonPath, "r")
Data = json.load(JsonObj)

JsonData = {}
JsonData["info"] = {}
JsonData["licenses"] = []

ImageIdVector = []
for index in range(ExtractRange[0], ExtractRange[1]+1):
    ImageIdVector.append(index)
if Shuffle:
    random.shuffle(ImageIdVector)

Images = []
Annotations = []
for index in range(len(ImageIdVector)):
    ImageId = ImageIdVector[index]
    image = Data["images"][ImageId].copy()
    image["id"] = index
    Images.append(image)

    for AnnotationsIndex in range(ImageId,len(Data["annotations"])-1):
        if Data["annotations"][AnnotationsIndex]["image_id"] == ImageId:
            annotation = Data["annotations"][AnnotationsIndex].copy()
            annotation["image_id"] = index
            Annotations.append(annotation)
            if Data["annotations"][AnnotationsIndex + 1]["image_id"] != ImageId:
                break

JsonData["images"] = Images
JsonData["annotations"] = Annotations

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
JsonData["categories"] = Categories

JsonObject = json.dumps(JsonData, indent=4)
# Writing to sample.json
with open(OutputJson, "w") as outfile:
    outfile.write(JsonObject)
