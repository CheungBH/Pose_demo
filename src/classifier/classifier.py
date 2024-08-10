from .ML.ML_classifier import MLClassifier
from .image.image_classifier import ImageClassifier
from .ML.seqML_classifier import SeqMLClassifier

import numpy as np
import cv2


class EnsembleClassifier:
    def __init__(self, types, weights, configs, labels, kps_transform, device="cuda:0"):
        self.classifiers = []
        if len(types) > 0:
            assert len(types) == len(weights) == len(configs) == len(labels), "Length of classifiers must be equal"
        for t, w, c, l in zip(types, weights, configs, labels):
            if t == "ML":
                self.classifiers.append(MLClassifier(weight=w, config=c, label=l))
            elif t == "image":
                self.classifiers.append(ImageClassifier(weight=w, config=c, label=l, transform=kps_transform, device=device))
            else:
                raise NotImplementedError("Not support this type of classifier")
        if len(types) > 0:
            self.max_label_len = self.get_longest_label()

    def visualize(self, actions, ids, frame, h_interval=40):
        img = np.zeros_like(frame, dtype=np.uint8)
        if not actions:
            return img

        cv2.putText(img, "id", (10, 28), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        for h_idx, h_item in enumerate(actions):
            if h_idx == 0:
                color = (0, 0, 255)
                cv2.line(img, (0, 35), (img.shape[1], 35), color, 2)

            cv2.putText(img, "cls {}".format(h_idx), (0, 30 + h_interval * (h_idx+1)),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            for w_idx, (i, action) in enumerate(zip(ids, h_item)):
                if h_idx == 0:
                    cv2.line(img, (85, 0), (85, img.shape[0]), (0, 0, 255), 2)
                    cv2.putText(img, str(i), (8 * self.max_label_len * (w_idx+1) + 4 * self.max_label_len, 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                action_color = (0, 255, 0)
                if isinstance(self.classifiers[h_idx], ImageClassifier):
                    if "whole" in self.classifiers[h_idx].img_type:
                        action_color = (255, 0, 0)
                cv2.putText(img, action, (8 * self.max_label_len * (w_idx+1), 30 + h_interval * (h_idx+1)),
                            cv2.FONT_HERSHEY_PLAIN, 2, action_color, 2)
        return img

    def update(self, image, ids, boxes, kps, kps_exist):
        inputs = {"image": image, "ids": ids, "boxes": boxes, "kps": kps, "kps_exist": kps_exist}
        actions = []
        for classifier in self.classifiers:
            action = classifier(**inputs)
            actions.append(action)
        self.actions = actions
        return actions

    def get_longest_label(self):
        max_len = []
        for classifier in self.classifiers:
            max_len.append(max([len(label) for label in classifier.classes]))
        return max(max_len)

