import json
from pycocotools.coco import COCO


def filter_person_instances(json_cate, input_json, output_json):
    coco = COCO(input_json)
    person_id = coco.getCatIds(catNms=[json_cate])
    img_ids = coco.getImgIds(catIds=person_id)

    img_info_list = coco.loadImgs(ids=img_ids)

    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=person_id)
    ann_info_list = coco.loadAnns(ids=ann_ids)

    new_json_data = {
        "images": img_info_list,
        "annotations": ann_info_list,
        "categories": [{"id": 1, "name": f"{json_cate}", "supercategory": f"{json_cate}"}]
    }

    with open(output_json, 'w') as f:
        json.dump(new_json_data, f)


category = "person"
input_train_json = "/media/hkuit164/Backup/coco/annotations/instances_train2017.json"
output_train_json = "/media/hkuit164/Backup/coco/annotations/person_instances_train2017.json"
filter_person_instances(category, input_train_json, output_train_json)

input_val_json = "/media/hkuit164/Backup/coco/annotations/instances_val2017.json"
output_val_json = "/media/hkuit164/Backup/coco/annotations/person_instances_val2017.json"
filter_person_instances(category, input_val_json, output_val_json)
