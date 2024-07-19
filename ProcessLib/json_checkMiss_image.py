import os
import json
import argparse
import traceback


def parse_args():
    parser = argparse.ArgumentParser(description='RtspDataCollection')
    # general
    parser.add_argument('--JsonFile',
                        help="json waiting for being checked",
                        required=True,
                        type=str)
    parser.add_argument('--ImageFolder',
                        help="image waiting for being checked",
                        required=True,
                        type=str)
    parser.add_argument('--DelMissImage',
                        help="miss image waiting for being deleted",
                        action="store_true",
                        required=False)
    args = parser.parse_args()
    return args

args = parse_args()
JsonPath = args.JsonFile
ImagePath = args.ImageFolder
print("Whether del miss image json or not: ", args.DelMissImage)
if os.path.exists(JsonPath):
    JsonObj = open(JsonPath, "r")
    JsonData = json.load(JsonObj)

ImageList = os.listdir(ImagePath)
JsonList = []

def check_json_image(JsonData, ImageList):
    miss_image = []
    del_id = []
    for data in JsonData["images"]:
        fileName = data["file_name"].split('-')[-1]
        if fileName not in ImageList:
            miss_image.append(fileName)
            if args.DelMissImage == True:
                del_id.append(data["id"])
                # del data
                # print("del: ", fileName)
                continue
        else:
            JsonList.append(fileName)
            continue
    for annotation in JsonData["annotations"]:
        if annotation["image_id"] in del_id:
            # del annotation
            continue
        else:
            continue
    return miss_image, del_id

missList, delList= check_json_image(JsonData, ImageList)
# print("Image list: ", ImageList)
print("length of Image list: ", len(ImageList))
print("length of Json list: ", len(JsonList))
print("diff: ", set(ImageList).difference(set(JsonList)))

print("Image miss: ", missList)
print("delete id: ", delList)
print("Miss number: ", len(missList))


