from .utils import scale, crop
import numpy as np
from config.config import kps_conf
from .id_color import COLORS
from PIL import ImageColor


class Visualizer:
    def __init__(self, kps_num, bg_type="raw", det_label="", kps_thresh=kps_conf, kps_color_type="COCO", color_type="by_label"):
        self.IDV = IDVisualizer()
        self.color_type = color_type
        assert color_type in ["normal", "by_id", "by_label"], "Unsupported color type: {}".format(color_type)
        if kps_color_type == "COCO":
            self.KPV = KeyPointVisualizer(kps_num, "coco", thresh=kps_thresh)
        elif kps_color_type == "per_id":
            self.KPV = KeyPointVisualizerID(kps_num, thresh=kps_thresh)
        else:
            raise NotImplementedError("Unsupported keypoint color type: {}".format(kps_color_type))
        self.det_cls = "" if not det_label else [name.strip() for name in open(det_label).readlines()]
        self.BBV = BBoxVisualizer(self.det_cls)
        self.BallV = BallVisualizer()
        self.bg_type = bg_type
        assert bg_type in ["raw", "black"], "Unsupported background type: {}".format(bg_type)
        self.kps_color_type = kps_color_type
        # Define 10 differet colors
        self.colors = [(128, 128, 128), (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), (255, 0, 0),
                       (255, 255, 0), (0, 0, 255)]

    def visualize(self, image, ids, boxes, boxes_cls, kps, kps_scores):
        vis_img = np.full(image.shape, 0, dtype=np.uint8) if self.bg_type == "black" else image.copy()
        if len(ids) > 0:
            if self.color_type == "normal":
                # if self.bg_type == "raw":
                self.BBV.visualize(boxes, vis_img, boxes_cls)
                self.IDV.plot_bbox_id(self.get_id2bbox(ids, boxes), vis_img)
                if self.kps_color_type == "COCO":
                    self.KPV.visualize(vis_img, kps, kps_scores)
                else:
                    self.KPV.visualize(vis_img, ids, kps, kps_scores)
            elif self.color_type == "by_id":
                for i, (id, box, box_cls, kp, kp_score) in enumerate(zip(ids, boxes, boxes_cls, kps, kps_scores)):
                    color = self.colors[int(id.tolist())]
                    # if self.bg_type == "raw":
                    self.BBV.visualize([box], vis_img, [box_cls], color)
                    self.IDV.plot_bbox_id({id: box}, vis_img, color=(color, color))
                    self.KPV.visualize(vis_img,  kp.unsqueeze(dim=0), kp_score.unsqueeze(dim=0), color=color)
            else: # by label
                for i, (id, box, box_cls, kp, kp_score) in enumerate(zip(ids, boxes, boxes_cls, kps, kps_scores)):
                    try:
                        color = self.colors[int(box_cls.tolist())]
                    except:
                        color = self.colors[int(box_cls)]

                    self.BBV.visualize([box], vis_img, [box_cls], color)
                    self.IDV.plot_bbox_id({id: box}, vis_img, color=(color, color))
                    self.KPV.visualize(vis_img, kp.unsqueeze(dim=0).cpu(), kp_score.unsqueeze(dim=0).cpu(), color=color)
        return vis_img

    def get_labels(self):
        return self.BBV.labels

    @staticmethod
    def get_id2bbox(ids, boxes):
        return {i: box for i, box in zip(ids, boxes)}

    def get_image_cropped_img(self, boxes, kps, kps_scores, image, type="black", scaled_ratio=0, exists=[1,2]):
        assert type in ["black", "raw"], "The keypoint type should be black or raw"
        img = image.copy() if type == "raw" else np.full(image.shape, 0, dtype=np.uint8)
        self.KPV.visualize(img, kps, kps_scores)
        cropped_imgs = []
        for i, box in enumerate(boxes):
            cropped_imgs.append(crop(scale(img, box, scaled_ratio), img))
        return cropped_imgs


class BBoxVisualizer:
    def __init__(self, cls_names):
        self.cls_names = cls_names
        self.box_color = (0, 255, 0)
        self.label_color = (0, 0, 255)
        self.labels = []

    def visualize(self, bboxes, img, boxes_cls=[], color=[]):
        box_color = self.box_color if not color else color
        label_color = self.label_color if not color else color
        self.labels = []
        for idx, bbox in enumerate(bboxes):
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), box_color, 4)
            if len(self.cls_names) > 0 and len(boxes_cls) > 0:
                if self.cls_names:
                    curr_label = self.cls_names[int(boxes_cls[idx])]
                    self.labels.append(curr_label)
                img = cv2.putText(img, curr_label, (int(bbox[0]), int(bbox[1])),
                                  cv2.FONT_HERSHEY_PLAIN, 2, label_color, 2)


class IDVisualizer(object):
    def __init__(self):
        pass

    def plot_bbox_id(self, id2bbox, img, color=((255, 0, 0), (0, 0, 255)), id_pos="down", with_bbox=False):
        for idx, box in id2bbox.items():
            idx = int(idx)
            [x1, y1, x2, y2] = box
            if id_pos == "up":
                cv2.putText(img, "id{}".format(idx), (int((x1 + x2)/2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 4,
                            color[0], 4)
            else:
                cv2.putText(img, "id{}".format(idx), (int((x1 + x2)/2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 4,
                            color[0], 4)
            if with_bbox:
                img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color[1], 4)


    # def plot_skeleton_id(self, id2ske, img):
    #     for idx, kps in id2ske.items():
    #
    #         x = np.mean(np.array([item[0] for item in kps]))
    #         y = np.mean(np.array([item[1] for item in kps]))
    #         cv2.putText(img, "id{}".format(idx), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN,1,
    #                     colors["yellow"], 2)


import torch
import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
WHITE = (255, 255, 255)
PINK = (180, 105, 255)


coco_p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
coco_line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                   (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                   (77, 222, 255), (255, 156, 127),
                   (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

mpii_p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE,
                BLUE]
mpii_line_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, RED, RED, BLUE, BLUE]


class BallVisualizer:
    def __init__(self):
        self.color = (141, 49, 1)

    def visualize(self, img, ball):
        cv2.circle(img, (int(ball[0]), int(ball[1])), 10, self.color, -1)


class KeyPointVisualizer:
    def __init__(self, kps, dataset, thresh=0.1):
        self.kps = kps
        self.thresh = self.process_thresh(thresh)
        if kps == 13:
            self.l_pair = [
                (1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
                (13, 7), (13, 8), (0, 13),  # Body
                (7, 9), (8, 10), (9, 11), (10, 12)
            ]
            self.p_color = coco_p_color
            self.line_color = coco_line_color

        elif kps == 17:
            self.l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                (17, 11), (17, 12),  # Body
                (11, 13), (12, 14), (13, 15), (14, 16)
            ]
            self.p_color = coco_p_color
            self.line_color = coco_line_color

        else:
            if dataset == "mpii":
                self.l_pair = [[0, 1], [1, 2], [2, 6], [6, 3], [3, 4], [4, 5], [6, 7], [7, 8], [8, 9], [8, 12],
                               [12, 11], [11, 10], [8, 13], [13, 14], [14, 15]]
                self.p_color = mpii_p_color
                self.line_color = mpii_line_color
            elif dataset == "aic":
                self.l_pair = [[2, 1], [1, 0], [0, 13], [13, 3], [3, 4], [4, 5], [8, 7], [7, 6], [6, 9], [9, 10],
                               [10, 11], [12, 13], [0, 6], [3, 9]]
                self.p_color = mpii_p_color
                self.line_color = mpii_line_color
            else:
                raise NotImplementedError("Not a suitable dataset and kps num!")

    def process_thresh(self, thresh):
        if isinstance(thresh, float):
            thresh = [thresh for _ in range(self.kps)]
            thresh = torch.Tensor(thresh)
        return thresh

    def visualize(self, frame, kps, kps_confs=[], color=()):
        kps = torch.Tensor(kps)
        if len(kps_confs) <= 0:
            kps_confs = torch.Tensor([[[1 for _ in range(kps.shape[0])] for j in range(kps.shape[1])]])
        else:
            kps_confs = torch.Tensor(kps_confs)

        for idx in range(len(kps)):
            part_line = {}
            kp_preds = kps[idx]
            kp_confs = kps_confs[idx]

            kp_thresh = self.thresh
            if self.kps == 17:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
                kp_confs = torch.cat((kp_confs, torch.unsqueeze((kp_confs[5, :] + kp_confs[6, :]) / 2, 0)))
                kp_thresh = torch.cat((self.thresh, torch.unsqueeze((self.thresh[5] + self.thresh[6]) / 2, 0)))
            elif self.kps == 13:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[1, :] + kp_preds[2, :]) / 2, 0)))
                kp_confs = torch.cat((kp_confs, torch.unsqueeze((kp_confs[1, :] + kp_confs[2, :]) / 2, 0)))
                kp_thresh = torch.cat((self.thresh, torch.unsqueeze((self.thresh[1] + self.thresh[2]) / 2, 0)))
            # Draw keypoints
            for n in range(kp_preds.shape[0]):
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                cor_conf = kp_confs[n, 0]

                if cor_x == 0 or cor_y == 0 or cor_conf < kp_thresh[n]:
                    continue

                part_line[n] = (cor_x, cor_y)
                p_color = self.p_color[n] if not color else color
                cv2.circle(frame, (cor_x, cor_y), 4, p_color, -1)
            # Draw limbs
            for i, (start_p, end_p) in enumerate(self.l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    line_color = self.line_color[i] if not color else color
                    cv2.line(frame, start_xy, end_xy, line_color, 8)


class KeyPointVisualizerID:
    def __init__(self, kps, thresh=0.1):
        assert kps == 13 or kps == 17, "Only support 13 or 17 keypoints"
        self.kps = kps
        self.thresh = self.process_thresh(thresh)
        self.colors = COLORS
        self.l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]

    def process_thresh(self, thresh):
        if isinstance(thresh, float):
            thresh = [thresh for _ in range(self.kps)]
            thresh = torch.Tensor(thresh)
        return thresh

    def visualize(self, frame, ids, kps, kps_confs=[]):
        kps = torch.Tensor(kps)
        if len(kps_confs) <= 0:
            kps_confs = torch.Tensor([[[1 for _ in range(kps.shape[0])] for j in range(kps.shape[1])]])
        else:
            kps_confs = torch.Tensor(kps_confs)

        for idx, (i, kp) in enumerate(zip(ids, kps)):
            color_idx = int(i.tolist())-1 if i < len(self.colors) else -1
            part_line = {}
            kp_preds = kps[idx]
            kp_confs = kps_confs[idx]

            kp_thresh = self.thresh
            if self.kps == 17:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[5, :] + kp_preds[6, :]) / 2, 0)))
                kp_confs = torch.cat((kp_confs, torch.unsqueeze((kp_confs[5, :] + kp_confs[6, :]) / 2, 0)))
                kp_thresh = torch.cat((self.thresh, torch.unsqueeze((self.thresh[5] + self.thresh[6]) / 2, 0)))
            elif self.kps == 13:
                kp_preds = torch.cat((kp_preds, torch.unsqueeze((kp_preds[1, :] + kp_preds[2, :]) / 2, 0)))
                kp_confs = torch.cat((kp_confs, torch.unsqueeze((kp_confs[1, :] + kp_confs[2, :]) / 2, 0)))
                kp_thresh = torch.cat((self.thresh, torch.unsqueeze((self.thresh[1] + self.thresh[2]) / 2, 0)))
            # Draw keypoints
            for n in range(kp_preds.shape[0]):
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                cor_conf = kp_confs[n, 0]

                if cor_x == 0 or cor_y == 0 or cor_conf < kp_thresh[n]:
                    continue
                # ImageColor.getcolor(self.colors[color_idx][idx], "RGB")
                part_line[n] = (cor_x, cor_y)
                cv2.circle(frame, (cor_x, cor_y), 4,
                           ImageColor.getcolor(self.colors[color_idx][idx], "RGB"), 50)
            # Draw limbs
            for i, (start_p, end_p) in enumerate(self.l_pair):
                if start_p in part_line and end_p in part_line:
                    start_xy = part_line[start_p]
                    end_xy = part_line[end_p]
                    cv2.line(frame, start_xy, end_xy, ImageColor.getcolor(self.colors[color_idx][0], "RGB"), 5)

