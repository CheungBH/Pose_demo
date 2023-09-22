import json

KeypointJson = "/media/hkuit164/Backup/ThermalProject/HRNet-Human-Pose-Estimation/ProcessLib/ThermalJson/20221025.json"
OutputJson = "/media/hkuit164/Backup/ThermalProject/HRNet-Human-Pose-Estimation/ProcessLib/ThermalJson/20221025Modified.json"

KeypointJsonObj = open(KeypointJson, "r")
KeypintData = json.load(KeypointJsonObj)

JsonData = {}
JsonData["info"] = {}
JsonData["licenses"] = []

images=[]
for Oject in KeypintData["images"]:
    image = {}
    image["file_name"] = Oject["file_name"]
    image["id"] = Oject["id"]
    image["height"] = Oject["width"]
    image["width"] = Oject["height"]
    images.append(image)
JsonData["images"] = images

annotations = []
for ann in KeypintData["annotations"]:
    annotation = {}

    imageId = ann["image_id"]
    width = KeypintData["images"][imageId]["height"]
    height = KeypintData["images"][imageId]["width"]

    annotation["bbox"] = [ann["bbox"][0]/height*width, ann["bbox"][1]/width*height, ann["bbox"][2]/height*width, ann["bbox"][3]/width*height]

    keypoints = []
    for i in range(17):
        keypoints.extend([ann["keypoints"][i*3]/height*width, ann["keypoints"][i*3+1]/width*height, ann["keypoints"][i*3+2]])
    annotation["keypoints"] = keypoints

    annotation["image_id"] = imageId
    annotation["id"] = ann["id"]
    annotation["area"] = ann["area"]
    annotation["category_id"] =  ann["category_id"]
    annotation["iscrowd"] = ann["iscrowd"]
    annotation["num_keypoints"] = ann["num_keypoints"]
    annotations.append(annotation)


JsonData["annotations"] = annotations


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
