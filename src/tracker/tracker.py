from .sort import Sort


class PersonTracker:
    def __init__(self):
        self.tracker = Sort()

    def track(self, boxes):
        id2box_tmp = self.tracker.update(boxes)
        id2box = {int(box[4]): box[:4] for box in id2box_tmp}
        return id2box
