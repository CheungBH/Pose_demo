
import cv2
import numpy as np
import torch


image_normalize_mean = [0.485, 0.456, 0.406]
image_normalize_std = [0.229, 0.224, 0.225]


def read_labels(label_path):
    with open(label_path, "r") as f:
        return [item[:-1] for item in f.readlines()]


def image_normalize(img_name, size=224):
    if isinstance(img_name, str):
        image_array = cv2.imread(img_name)
    else:
        image_array = img_name
    image_array = cv2.resize(image_array, (size, size))
    image_array = np.ascontiguousarray(image_array[..., ::-1], dtype=np.float32)
    image_array = image_array.transpose((2, 0, 1))
    for channel, _ in enumerate(image_array):
        image_array[channel] /= 255.0
        image_array[channel] -= image_normalize_mean[channel]
        image_array[channel] /= image_normalize_std[channel]
    image_tensor = torch.from_numpy(image_array).float()
    return image_tensor


def get_pretrain(model_path):
    if "_resnet18" in model_path:
        name = "resnet18"
    elif "_resnet50" in model_path:
        name = "resnet50"
    elif "_resnet34" in model_path:
        name = "resnet34"
    elif "_resnet101" in model_path:
        name = "resnet101"
    elif "_resnet152" in model_path:
        name = "resnet152"
    elif "_inception" in model_path:
        name = "inception"
    elif "_mobilenet" in model_path:
        name = "mobilenet"
    elif "_shufflenet" in model_path:
        name = "shufflenet"
    elif "_LeNet" in model_path:
        name = "LeNet"
    elif "_squeezenet" in model_path:
        name = "squeezenet"
    elif "mnasnet" in model_path:
        name = "mnasnet"
    elif "LeNet" in model_path:
        name = "LeNet"
    else:
        raise ValueError("Wrong name of pre-train model")
    return name


def crop(box, img):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    x1 = 0 if x1 < 0 else x1
    y1 = 0 if y1 < 0 else y1
    x2 = img.shape[1] if x2 > img.shape[1] else x2
    y2 = img.shape[0] if y2 > img.shape[0] else y2
    cropped_img = np.asarray(img[y1: y2, x1: x2])
    return cropped_img


def scale(img, bbox, sf):
    assert len(sf) == 4, "You should assign 4 different factors value"
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    width, height = x_max - x_min, y_max - y_min
    imgheight = img.shape[0]
    imgwidth = img.shape[1]
    x_enlarged_min = max(0, x_min - width * sf[0])
    y_enlarged_min = max(0, y_min - height * sf[1])
    x_enlarged_max = min(imgwidth - 1, x_max + width * sf[2])
    y_enlarged_max = min(imgheight - 1, y_max + height * sf[3])
    return [x_enlarged_min, y_enlarged_min, x_enlarged_max, y_enlarged_max]