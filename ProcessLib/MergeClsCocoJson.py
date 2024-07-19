import os
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import time

def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--JsonFolder',
                        help="json waiting for being merged",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args
time_start = time.time()
args = parse_args()

MeragePath = args.JsonFolder
OutputJson = os.path.join(MeragePath, "MergedClsCoco.json")

MerageFolders = sorted(os.listdir(MeragePath))

DataList = []
for Obj in MerageFolders:
    if Obj[0] == ".":
        continue
    JsonPath = os.path.join(MeragePath, Obj)
    JsonObj = open(JsonPath, "r")
    Data = json.load(JsonObj)
    DataList.append(Data)

ImageIdCounter = 0
AnnotationIdCounter = 0
MeragedImages = []
meragedAnnotation = []

def handle_image(JsonData):
    images = []
    for Image in JsonData["images"]:
        image = {}

        split_list = Image["file_name"].split('-')
        Image["file_name"] = split_list[-1]
        # print("after: ", Image["file_name"])
        image["file_name"] = Image["file_name"]
        # print("upload: ", image["file_name"])

        image["id"] = Image["id"] + ImageIdCounter
        image["height"] = Image["height"]
        image["width"] = Image["width"]
        images.append(image)
    return images

def regulate_category(name):
    name_to_id = {
        "backhand": 0,
        "Backhand": 0,
        "forehand": 1,
        "Forehand": 1,
        "overhead": 2,
        "Overhead": 2,
        "waiting": 3,
        "Waiting": 3,
        "others": 3,
        "Others": 3,
        "other": 3,
        "Other": 3
    }
    return name_to_id.get(name)

def handle_annotation(JsonData, category):
    annotations = []

    for Annotations in JsonData["annotations"]:
        annotation = {}
        annotation["bbox"] = Annotations["bbox"]
        annotation["segmentation"] = Annotations["segmentation"]
        annotation["ignore"] = Annotations["ignore"]
        annotation["image_id"] = Annotations["image_id"] + ImageIdCounter
        annotation["id"] = Annotations["id"] + AnnotationIdCounter
        annotation["area"] = Annotations["area"]

        if JsonData["categories"] != category:
            for categories in JsonData["categories"]:
                if categories["id"] == Annotations["category_id"]:
                    annotation["category_id"] = regulate_category(categories["name"])
        else:
            annotation["category_id"] = Annotations["category_id"]

        annotation["iscrowd"] = Annotations["iscrowd"]
        annotations.append(annotation)
    return annotations

# Categories = []
Categories = [
    {
      "id": 0,
      "name": "backhand"
    },
    {
      "id": 1,
      "name": "forehand"
    },
    {
      "id": 2,
      "name": "overhead"
    },
    {
      "id": 3,
      "name": "waiting"
    }
  ]
# Categories.append(Category)

with tqdm(DataList) as pbar:
    print("---------------------------------------Merge begin------------------------------------------")
    for index, JsonData in enumerate(pbar):
        if index == 0:

            images = handle_image(JsonData)
            MeragedImages.extend(images)

            first_annotations = handle_annotation(JsonData, Categories)
            meragedAnnotation.extend(first_annotations)

            ImageIdCounter = len(JsonData["images"])
            AnnotationIdCounter = len(JsonData["annotations"])
            continue

        images = handle_image(JsonData)
        MeragedImages.extend(images)

        annotations = handle_annotation(JsonData, Categories)
        meragedAnnotation.extend(annotations)

        ImageIdCounter = len(MeragedImages)
        AnnotationIdCounter = len(meragedAnnotation)


create_date_and_time = datetime.now()
New_JsonData = {}
New_JsonData["info"] = {
    "year": 2024,
    "version": "1.0",
    "description": "",
    "contributor": "Label Studio",
    "url": "",
    "date_created": f"{create_date_and_time}"
}
New_JsonData["images"] = MeragedImages
New_JsonData["annotations"] = meragedAnnotation
New_JsonData["categories"] = Categories

JsonObject = json.dumps(New_JsonData, indent=4)
# Writing to sample.json
with open(OutputJson, "w") as outfile:
    outfile.write(JsonObject)

time_end = time.time()
print("Total time cost: ", time_end-time_start)
print("---------------------------------------Merge end---------------------------------------------")
