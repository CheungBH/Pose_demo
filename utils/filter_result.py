
class ResultFilterer:
    def __init__(self, criterion):
        self.filter_list = {"time": self.filter_by_time, "id": self.filter_by_id,
                            "boxes": self.filter_by_boxes, "kps": self.filter_by_kps}
        self.filter_criterion = criterion

    def filter_by_time(self, cnt):
        if "time" in self.filter_criterion:
            for begin_frame, end_frame in self.filter_criterion["time"]:
                if begin_frame < cnt < end_frame:
                    return True
            return False
        else:
            return True

    def filter_by_id(self, idx):
        if "id" in self.filter_criterion:
            if idx in self.filter_criterion["id"]:
                return True
            return False
        return True

    def filter_by_boxes(self):
        return True

    def filter_by_kps(self):
        return True

    def filter(self, idx, boxes, kps, kps_scores, cnt):
        if not self.filter_criterion:
            return idx, boxes, kps, kps_scores
        save_idx = []
        for i in range(len(idx)):
            if self.filter_by_time(cnt) and self.filter_by_id(i) and self.filter_by_boxes() and self.filter_by_kps():
                save_idx.append(i)
        return idx[save_idx], boxes[save_idx], kps[save_idx], kps_scores[save_idx]
