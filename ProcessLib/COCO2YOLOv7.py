import json
import os
import shutil


def convert_coco_to_yolo(coco_annotation_file, output_dir, img_dir):
    # Load COCO annotation JSON file
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = {image['id']: image for image in coco_data['images']}
    categories = {category['id']: category['name'] for category in coco_data['categories']}

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Process each annotation in the COCO dataset
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        # COCO bbox format: [x_min, y_min, width, height]
        x_min, y_min, width, height = bbox
        image_info = images[image_id]
        img_width = image_info['width']
        img_height = image_info['height']

        # YOLO bbox format: [x_center, y_center, width, height] normalized
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        width /= img_width
        height /= img_height

        category_name = categories[category_id]
        category_index = list(categories.values()).index(category_name)

        yolo_annotation = f"{category_index} {x_center} {y_center} {width} {height}\n"

        # Write YOLO format annotation to the corresponding .txt file
        image_filename = image_info['file_name']
        txt_filename = os.path.splitext(image_filename)[0] + '.txt'
        txt_filepath = os.path.join(output_dir, txt_filename)

        with open(txt_filepath, 'a') as f:
            f.write(yolo_annotation)

    # Copy images to the output directory (optional)
    for image in coco_data['images']:
        src_path = os.path.join(img_dir, image['file_name'])
        # dst_path = os.path.join(output_dir, image['file_name'])
        if os.path.exists(src_path):
            # shutil.copy(src_path, dst_path)
            continue
        else:
            print(f"Warning: Image {src_path} not found.")


if __name__ == '__main__':
    coco_annotation_file = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_sources/pose_sources_0710/label_studio_concise/MergedCoco.json'
    output_dir = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_sources/pose_sources_0710/YOLOv7_trainable'
    img_dir = '/media/hkuit164/WD20EJRX/Aiden/Tennis_dataset/pose_sources/pose_sources_0710/images/total_images'
    convert_coco_to_yolo(coco_annotation_file, output_dir, img_dir)
