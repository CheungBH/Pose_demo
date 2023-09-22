import numpy as np


def crop(box, img):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = img.shape[1] if x2 > img.shape[1] else x2
    y2 = img.shape[0] if y2 > img.shape[0] else y2
    cropped_img = np.asarray(img[y1: y2, x1: x2])
    return cropped_img


def scale(img, bbox, sf=0):
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    width, height = x_max - x_min, y_max - y_min
    imgheight = img.shape[0]
    imgwidth = img.shape[1]
    x_enlarged_min = max(0, x_min - width * sf / 2)
    y_enlarged_min = max(0, y_min - height * sf / 2)
    x_enlarged_max = min(imgwidth - 1, x_max + width * sf / 2)
    y_enlarged_max = min(imgheight - 1, y_max + height * sf / 2)
    return [x_enlarged_min, y_enlarged_min, x_enlarged_max, y_enlarged_max]
