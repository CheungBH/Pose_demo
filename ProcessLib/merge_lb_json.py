import os
import json


def merge_json_files(input_folder, output_file):
    merged_data = {
        "images": [],
        "categories": [],
        "annotations": [],
        "info": {}
    }

    image_id_offset = 0
    annotation_id_offset = 0

    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # Update images
                for image in data.get('images', []):
                    image['id'] += image_id_offset
                    merged_data['images'].append(image)
                image_id_offset = len(merged_data['images'])

                # Update annotations
                for annotation in data.get('annotations', []):
                    annotation['id'] += annotation_id_offset
                    annotation['image_id'] += image_id_offset
                    merged_data['annotations'].append(annotation)
                annotation_id_offset = len(merged_data['annotations'])

                # Update categories
                for category in data.get('categories', []):
                    if category not in merged_data['categories']:
                        merged_data['categories'].append(category)

                # Update info
                merged_data['info'].update(data.get('info', {}))

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


input_folder = '/media/hkuit164/WD20EJRX/ESTRNN_dataset/relabel/total_json'
output_file = '/media/hkuit164/WD20EJRX/ESTRNN_dataset/relabel/total_json.json'
merge_json_files(input_folder, output_file)
