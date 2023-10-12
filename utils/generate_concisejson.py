import json
import config as config

class ConciseJsonGenerator:
    def __init__(self, json_name):
        if not json_name:
            json_name = "result.json"
        self.file = open(json_name, "w")
        self.save_image = 0
        self.image_count = 0
        self.anno_count = 0

        self.JsonData = {}
        self.JsonData["info"] = {}
        self.JsonData["licenses"] = []
        self.images=[]
        self.JsonData["images"] = self.images
        self.JsonData["annotations"] = []
        self.JsonData["categories"] = []
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
        self.JsonData["categories"].append(Category)


    def update(self, idx, boxes, kps, kps_scores, cnt, img_h, img_w):

        # If no bounding boxes, do not save frame
        if len(idx) == 0:
            self.save_image = 0
            return

        self.save_image = 1
        idx, boxes, kps, kps_scores = idx.tolist(), boxes.tolist(), kps.tolist(), kps_scores.tolist()
        for i, box, kp, kp_score in zip(idx, boxes, kps, kps_scores):
            image = {}
            image["file_name"] = str(cnt)+".jpg"
            image["id"] = self.image_count
            image["height"] = config.out_size[1]
            image["width"] = config.out_size[0]
            self.images.append(image)

            item = {}
            item["bbox"] = box

            keypoints =[]
            for Keypoint in kp:
                Keypoint[0] = Keypoint[0]
                Keypoint[1] = Keypoint[1]

                keypoints.extend(Keypoint)
                keypoints.extend([2])
            item["keypoints"] = keypoints

            item["image_id"] = self.image_count
            item["id"] = self.anno_count
            self.anno_count += 1
            item["area"] = box[2] * box[3]
            item["category_id"] =  1
            item["iscrowd"] = 0
            item["num_keypoints"] = 17
            self.JsonData["annotations"].append(item)

        self.image_count += 1



    def release(self):
        self.file.write(json.dumps(self.JsonData, indent=2))
