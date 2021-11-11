from .sort import Sort


class PersonTracker:
    def __init__(self):
        self.tracker = Sort()

    def track(self, boxes):
        self.ids, self.boxes = [], []
        tracked_boxes = self.tracker.update(boxes)
        for tracked_box in tracked_boxes:
            self.ids.append(int(tracked_box[4]))
            self.boxes.append(tracked_box[:4])
        return self.ids, self.boxes

    def get_id2bbox(self):
        return {i: box for i, box in zip(self.ids, self.boxes)}
