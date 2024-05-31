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
import glob


def read_content(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

txt_folder = ""
output_folder = ""
classe_name = ["Stand", "Swim", "Warning"]

os.makedirs(output_folder, exist_ok=True)
txt_files = glob.glob(os.path.join(txt_folder, '*.txt'))

width = 1280
height = 720
instance_idx = 0

# Image input folder path
storage = r"C:\Users\hkuit\AppData\Local\label-studio\label-studio\media\upload"
# project number
project_number = sys.argv[1]
storage = os.path.join(storage, project_number)


# Loop through the images in the folder
for filename in os.listdir(storage):
    if filename.endswith((".jpg")):

        # Image file path
        image_path = os.path.join("\\data\\upload", project_number, filename)

        # Extract the desired part of the filename
        # Label-studio add randomly generated prefix that ends with "-"
        # for all uploaded images to prevent duplication.
        prefix = "-"
        suffix = os.path.splitext(filename)[0]
        extracted_part = suffix.split(prefix, 1)[-1]
        image_name = extracted_part + ".jpg"
        txt_file = os.path.join(txt_folder, extracted_part + ".txt")

        data = {
            "data": {
                "image": image_path
            },
            "annotations": [{
                "result": []
            }]
        }

        txt_content = read_content(txt_file)
        for content in txt_content:
            cls, x, y, w, h = content.split(" ")
            x, y, w, h = float(x), float(y), float(w), float(h),
            x = x - w/2
            y = y - h/2
            # center_x, center_y = (x1+x2)/2, (y1+y2)/2
            # h, w = y2-y1, x2-x1
            rectangle = {
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                # label-studio uses 0-100 normalization
                "value": {

                    "x": x*width*(100/width),
                    "y": y*height*(100/height),
                    "width": w*width*(100/width),
                    "height": h*height*(100/height),

                    "rotation": 0,
                    # dummy labels that will be change using label-studio.
                    "rectanglelabels": [
                        classe_name[int(cls)]
                    ]
                },
                "id": instance_idx,
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "origin": "manual"
            }
            data["annotations"][0]["result"].append(rectangle)
            instance_idx += 1

    output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".json")
    with open(output_file, "w") as json_file:
        json.dump(data, json_file, indent=2)
