from .box_sort.sort_tracker import BoxTracker
from .deep_sort.deep_sort import DeepSort
import torch
import cv2


class PersonTracker:
    def __init__(self, sort_type, device="cuda:0", **kwargs):
        if sort_type == "sort":
            self.tracker = BoxTracker()
        elif sort_type == "deepsort":
            use_cuda = True if device != "cpu" else False
            self.tracker = DeepSort(use_cuda=use_cuda, **kwargs)
        else:
            raise NotImplementedError("sort type {} not implemented".format(sort_type))

    def update(self, boxes, ori_img):
        self.ids, self.boxes = [], []
        boxes = self.box_xyxy2xywh(boxes) if isinstance(self.tracker, DeepSort) else boxes
        conf = torch.ones(len(boxes))
        if isinstance(self.tracker, DeepSort):
            tracked_boxes = self.tracker.update(boxes, conf, ori_img)
        else:
            tracked_boxes = self.tracker.update(boxes)
        for tracked_box in tracked_boxes:
            self.ids.append(int(tracked_box[4]))
            self.boxes.append(tracked_box[:4])
        return self.ids, self.boxes

    def box_xyxy2xywh(self, boxes):
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
        return boxes[:, :4]

    def get_id2bbox(self):
        return {i: box for i, box in zip(self.ids, self.boxes)}

    def get_pred(self):
        if isinstance(self.tracker, DeepSort):
            return {}
        return self.tracker.tracker.id2pred

    def plot_iou_map(self, img, h_interval=40, w_interval=80):
        if isinstance(self.tracker, DeepSort):
            return img
        tracker = self.tracker.tracker
        iou_matrix = tracker.mat
        match_pairs = [(pair[0], pair[1]) for pair in tracker.match_indices]
        if len(iou_matrix) > 0:
            for h_idx, h_item in enumerate(iou_matrix):
                if h_idx == 0:
                    color = (0, 0, 255)
                    cv2.line(img, (0, 35), (img.shape[1], 35), color, 2)

                for w_idx, item in enumerate(h_item):
                    if w_idx == 0 or h_idx == 0:
                        color = (0, 0, 255)
                        if h_idx == 0:
                            cv2.line(img, (80, 0), (80, img.shape[0]), color, 2)
                    elif (w_idx - 1, h_idx - 1) in match_pairs:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    cv2.putText(img, item, (-65 + w_interval * w_idx, 30 + h_interval * h_idx), cv2.FONT_HERSHEY_PLAIN,
                                2, color, 2)
        return img



