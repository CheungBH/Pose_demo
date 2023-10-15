from .ML.ML_classifier import MLClassifier
from .image.image_classifier import ImageClassifier


class EnsembleClassifier:
    def __init__(self, types, weights, configs, labels, device="cuda:0"):
        self.classifiers = []
        for t, w, c, l in zip(types, weights, configs, labels):
            if t == "ML":
                self.classifiers.append(MLClassifier(weight=w, config=c, label=l))
            elif t == "image":
                self.classifiers.append(ImageClassifier(weight=w, config=c, label=l, device=device))
            else:
                raise NotImplementedError("Not support this type of classifier")

    def visualize(self):
        pass

    def update(self, image, box, kps, kps_exist):
        actions = []
        for classifier in self.classifiers:
            action = classifier(img=image, box=box, kps=kps)
            actions.append(action)
        self.actions = actions


