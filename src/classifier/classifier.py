from .ML.ML_classifier import MLClassifier
from .image.image_classifier import ImageClassifier


class EnsembleClassifier:
    def __init__(self, types, weights, configs, labels, kps_transform, device="cuda:0"):
        self.classifiers = []
        for t, w, c, l in zip(types, weights, configs, labels):
            if t == "ML":
                self.classifiers.append(MLClassifier(weight=w, config=c, label=l))
            elif t == "image":
                self.classifiers.append(ImageClassifier(weight=w, config=c, label=l, transform=kps_transform, device=device))
            else:
                raise NotImplementedError("Not support this type of classifier")

    def visualize(self):
        pass

    def update(self, image, boxes, kps, kps_exist):
        actions = []
        for classifier in self.classifiers:
            action = classifier(img=image, boxes=boxes, kps=kps, kps_exist=kps_exist)
            actions.append(action)
        self.actions = actions


