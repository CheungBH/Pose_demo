
class ResultFilterer:
    def __init__(self, criterion):
        self.filter_list = {"time": self.filter_by_time, "id": self.filter_by_id,
                            "boxes": self.filter_by_boxes, "kps": self.filter_by_kps}
        self.filter_criterion = criterion

    def filter_by_time(self):
        pass

    def filter_by_id(self):
        pass

    def filter_by_boxes(self):
        pass

    def filter_by_kps(self):
        pass

    def filter(self, id, boxes, kps, kps_scores, cnt):
        self.id, self.boxes, self.kps, self.kps_scores, self.cnt = id, boxes, kps, kps_scores, cnt
