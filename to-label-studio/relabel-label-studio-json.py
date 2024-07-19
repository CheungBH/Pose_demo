import os
import json
import sys

# README
# Start a new project in label-studio and upload the images.
# Check the image storage of label-studio at
# cd .local/share/label-studio/media/upload
# get "project_id" of where you import the images for running of this script.
# run script: 
# python create_json.py project_id label-studio-concise.json
# e.g. python create_json.py 2 /full/path/result_project_15.json

# Get the path to the user's home directory
home_dir = os.path.expanduser("~")
output_folder = "/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/cls_sources/cls_sources_0710/relabel_cls_json"
os.makedirs(output_folder, exist_ok=True)

# Image input folder path
storage = os.path.join(home_dir, ".local/share/label-studio/media/upload")
# project number
project_number = sys.argv[1]
storage = os.path.join(storage, project_number)

# relabel_dict = {"Overhead": "overhead"}

with open(sys.argv[2], "r") as f:
    source_json = json.load(f)

# categories = {}
categories = source_json["categories"]
# for i in source_json["categories"]:
#     categories[i["id"]] = i["name"]

# Loop through the images in the folder
for filename in os.listdir(storage):
    if filename.endswith((".jpg")):
        # Image file path
        image_path = os.path.join("/data/upload", project_number, filename)

        # Extract the desired part of the filename
        # Label-studio add randomly generated prefix that ends with "-"
        # for all uploaded images to prevent duplication.
        prefix = "-"
        suffix = os.path.splitext(filename)[0]
        extracted_part = suffix.split(prefix, 1)[-1]
        image_name = extracted_part + ".jpg"




        # get image_id corrsponding to the image name.
        for image in source_json["images"]:
            if image["file_name"].split("-")[-1] == image_name.split("-")[-1]:
                image_id = image["id"]
                height = image["height"]
                width = image["width"]
                
        # generate label-studio import structure.
        data = {
            "data": {
                "image": image_path
            },
            "annotations": [{
                "result": []
            }]
        }

        # 
        for anno in source_json["annotations"]:
            cls = categories[anno["category_id"]]

            # updated_label = relabel_dict[cls] if cls in relabel_dict else cls
            # get every annotation of the same image.
            if anno["image_id"] == image_id:
    
                rectangle = {
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    # label-studio uses 0-100 normalization
                    "value": {

                        "x": anno["bbox"][0]*(100/width),
                        "y": anno["bbox"][1]*(100/height),
                        "width": anno["bbox"][2]*(100/width),
                        "height": anno["bbox"][3]*(100/height),

                        "rotation": 0,
                        # dummy labels that will be change using label-studio.
                        "rectanglelabels": [
                            # updated_label
                            cls['name']
                        ]
                    },
                    "id": anno["id"],
                    "from_name": "label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": "manual"
                }
                data["annotations"][0]["result"].append(rectangle)

        # Generate the JSON file
        output_file = os.path.join(output_folder, "-".join(os.path.splitext(filename)[0].split("-")[1:]) + ".json")
        with open(output_file, "w") as json_file:
            json.dump(data, json_file, indent=2)
        a =1
