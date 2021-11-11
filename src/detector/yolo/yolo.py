from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets import *
from .utils.utils import *

device = "cuda:0"


class YoloDetector:
    def __init__(self, cfg, weight, img_size=416, conf_thresh=0.4, nms_thresh=0.5):
        self.img_size = img_size
        self.model = Darknet(cfg, img_size)
        if weight.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weight, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, weight)

        self.model.to(device).eval()
        self.conf = conf_thresh
        self.nms = nms_thresh

    def inference(self, img):
        original_shape = img.shape
        with torch.no_grad():
            img = letterbox(img, new_shape=self.img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
            img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to fp16/fp32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred, _ = self.model(img)
            dets = non_max_suppression(pred, self.conf, self.nms)
            for det in dets:
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_shape).round()
            if dets[0] is None:
                return []
            return dets[0].cpu()


if __name__ == '__main__':
    import cv2
    cfg = "/home/hkuit164/Downloads/yolo_selected/coco_basic/pytorch/yolov3-original-1cls-leaky.cfg"
    weight = "/home/hkuit164/Downloads/yolo_selected/coco_basic/pytorch/last.weights"

    detector = YoloDetector(cfg, weight)

    img_path = "/media/hkuit164/Elements/data/posetrack18/images/test/000693_mpii_test/000013.jpg"
    img = cv2.imread(img_path)
    dets = detector.inference(img)
    for bbox in dets:
        bbox = bbox[:4].tolist()
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 4)
    cv2.imshow("result", img)
    cv2.waitKey(0)

    # print(dets)


