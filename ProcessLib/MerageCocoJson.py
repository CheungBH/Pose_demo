import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--JsonFolder',
                        help="json waiting for being meraged",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()

MeragePath = args.JsonFolder
OutputJson = os.path.join(MeragePath, "MeagedCoco.json")

MerageFolders = sorted(os.listdir(MeragePath))

DataList = []
for Obj in MerageFolders:
    if Obj[0] == ".":
    	continue
    JsonPath = os.path.join(MeragePath,Obj)
    JsonObj = open(JsonPath, "r")
    Data = json.load(JsonObj)
    DataList.append(Data)

ImageIdCounter = 0
AnnotationIdCounter = 0
MeragedImages = []
meragedAnnotation = []
for index, JsonData in enumerate(DataList):
    if index == 0:
        ImageIdCounter = len(JsonData["images"])
        AnnotationIdCounter = len(JsonData["annotations"])
        MeragedImages.extend(JsonData["images"])
        meragedAnnotation.extend(JsonData["annotations"])
        continue

    images = []
    for Image in JsonData["images"]:
        image = {}
        image["file_name"] = Image["file_name"]
        image["id"] = Image["id"] + ImageIdCounter
        image["height"] = Image["height"]
        image["width"] = Image["width"]
        images.append(image)
    MeragedImages.extend(images)

    annotations = []
    for Annotations in JsonData["annotations"]:
        annotation = {}
        annotation["bbox"] = Annotations["bbox"]
        annotation["keypoints"] = Annotations["keypoints"]

        annotation["image_id"] = Annotations["image_id"] + ImageIdCounter
        annotation["id"] = Annotations["id"] + AnnotationIdCounter
        annotation["area"] = Annotations["area"]
        annotation["category_id"] = Annotations["category_id"]
        annotation["iscrowd"] = Annotations["iscrowd"]
        annotation["num_keypoints"] = Annotations["num_keypoints"]
        annotations.append(annotation)
    meragedAnnotation.extend(annotations)

    ImageIdCounter = len(MeragedImages)
    AnnotationIdCounter = len(meragedAnnotation)

JsonData = {}
JsonData["info"] = {}
JsonData["licenses"] = []
JsonData["images"] = MeragedImages
JsonData["annotations"] = meragedAnnotation


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
