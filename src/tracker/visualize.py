import cv2
import numpy as np


def plot_id_box(id2bbox, img, color=(0, 0, 255), id_pos="up"):
    for idx, box in id2bbox.items():
        [x1, y1, x2, y2] = box
        if id_pos == "up":
            cv2.putText(img, "id{}".format(idx), (int((x1 + x2) / 2), int(y1)), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
        else:
            cv2.putText(img, "id{}".format(idx), (int((x1 + x2) / 2), int(y2)), cv2.FONT_HERSHEY_PLAIN, 4,
                        color, 4)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 4)

