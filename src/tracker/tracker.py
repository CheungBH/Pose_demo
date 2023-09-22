from .sort import Sort
from config import max_age, min_hints, iou_thresh
import cv2


class PersonTracker:
    def __init__(self):
        self.tracker = Sort(max_age=max_age, min_hits=min_hints, iou_threshold=iou_thresh)

    def track(self, boxes):
        self.ids, self.boxes = [], []
        tracked_boxes = self.tracker.update(boxes)
        for tracked_box in tracked_boxes:
            self.ids.append(int(tracked_box[4]))
            self.boxes.append(tracked_box[:4])
        return self.ids, self.boxes

    def get_id2bbox(self):
        return {i: box for i, box in zip(self.ids, self.boxes)}

    def get_pred(self):
        return self.tracker.id2pred

    def plot_iou_map(self, img, h_interval=40, w_interval=80):
        iou_matrix = self.tracker.mat
        match_pairs = [(pair[0], pair[1]) for pair in self.tracker.match_indices]
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
                    elif (w_idx-1, h_idx-1) in match_pairs:
                        color = (0, 255, 0)
                    else:
                        color = (255, 0, 0)
                    cv2.putText(img, item, (-65+w_interval*w_idx, 30+h_interval*h_idx), cv2.FONT_HERSHEY_PLAIN,
                                2, color, 2)
        return img
