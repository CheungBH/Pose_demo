import os
import json
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--JsonPath',
                        help="coco json",
                        required=True,
                        type=str)
    parser.add_argument('--InputFolder',
                        help="Inuput's image's folder",
                        required=True,
                        type=str)
    parser.add_argument('--OutputFolder',
                        help="Selected images in Json",
                        default = False,
                        type=str)
    args = parser.parse_args()
    return args

args = parse_args()

JsonPath = args.JsonPath
FolderPath = args.InputFolder
OuputPath = args.OutputFolder
move = True

JsonObj = open(JsonPath, "r")
Data = json.load(JsonObj)

for imageObj in Data["images"]:
    source = os.path.join(FolderPath, imageObj["file_name"])
    destination = os.path.join(OuputPath, imageObj["file_name"])
    if move:
        shutil.move(source, destination)
    else:
        shutil.copyfile(source, destination)
