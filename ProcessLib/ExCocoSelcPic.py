import os
import json
import random
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Split JSON into train and val')
    parser.add_argument('--JsonPath', help="Path to the input JSON file", required=True, type=str)
    parser.add_argument('--TrainRatio', help="Ratio of data to be included in the train set", default=0.8, type=float)
    parser.add_argument('--OutputTrainJson', help="Path to the output train.json file", required=True, type=str)
    parser.add_argument('--OutputValJson', help="Path to the output val.json file", required=True, type=str)
    parser.add_argument('--InputFolder', help="Inuput's image's folder", required=True, type=str)
    parser.add_argument('--OutputTrainFolder', help="Selected images in Json", default=False, type=str)
    parser.add_argument('--OutputValFolder', help="Selected images in Json", default=False, type=str)
    return parser.parse_args()

args = parse_args()

json_path = args.JsonPath
train_ratio = args.TrainRatio
output_train_json = args.OutputTrainJson
output_val_json = args.OutputValJson
FolderPath = args.InputFolder
OutputTrainPath = args.OutputTrainFolder
OutputValPath = args.OutputValFolder
os.makedirs(OutputValPath, exist_ok=True)
os.makedirs(OutputTrainPath, exist_ok=True)


with open(json_path, "r") as json_file:
    data = json.load(json_file)

images = data["images"]
annotations = data["annotations"]

random.shuffle(images)

train_size = int(train_ratio * len(images))
train_images = images[:train_size]
val_images = images[train_size:]

train_image_mapping = {}
val_image_mapping = {}

train_annotations = []
val_annotations = []

for i, img in enumerate(train_images):
    train_image_mapping[img["id"]] = i
    img["id"] = i

for i, img in enumerate(val_images):
    val_image_mapping[img["id"]] = i
    img["id"] = i

for ann in annotations:
    if ann["image_id"] in train_image_mapping:
        ann["image_id"] = train_image_mapping[ann["image_id"]]
        train_annotations.append(ann)

    elif ann["image_id"] in val_image_mapping:
        ann["image_id"] = val_image_mapping[ann["image_id"]]
        val_annotations.append(ann)


train_json = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": train_images,
    "annotations": train_annotations,
    "categories": data.get("categories", [])
}

val_json = {
    "info": data.get("info", {}),
    "licenses": data.get("licenses", []),
    "images": val_images,
    "annotations": val_annotations,
    "categories": data.get("categories", [])
}

with open(output_train_json, "w") as train_file:
    json.dump(train_json, train_file, indent=4)

with open(output_val_json, "w") as val_file:
    json.dump(val_json, val_file, indent=4)


JsonTrain = open(output_train_json, "r")
DataTrain = json.load(JsonTrain)

for imageObjt in DataTrain["images"]:
    source = os.path.join(FolderPath, imageObjt["file_name"])
    destination = os.path.join(OutputTrainPath, imageObjt["file_name"])
    shutil.copyfile(source, destination)

JsonVal = open(output_val_json, "r")
DataVal = json.load(JsonVal)

for imageObjv in DataVal["images"]:
    source = os.path.join(FolderPath, imageObjv["file_name"])
    destination = os.path.join(OutputValPath, imageObjv["file_name"])
    shutil.copyfile(source, destination)
