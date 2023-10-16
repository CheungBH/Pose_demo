import joblib
import numpy as np


class MLClassifier:
    def __init__(self, weight, config, label):
        self.joblib_model = joblib.load(weight)
        self.label = label
        with open(self.label, 'r') as file:
            self.lines = file.readlines()
    def __call__(self, img, boxes, kps, kps_exist):
        actions = []

        for box, kp in zip(boxes, kps):
            img_w = box[2] - box[0]
            img_h = box[3] - box[1]
            float_numbers = [float(i) for i in kp.flatten().tolist()]

            modified_array = []
            for index, num in enumerate(float_numbers):
                if index % 2 == 0:
                    modified_array.append(num / img_w)
                else:
                    modified_array.append(num / img_h)

            predict_num = self.joblib_model.predict([np.array(modified_array)])

            predict_action = self.lines[int(predict_num)]
            actions.append(predict_action)
        return actions

    def preprocess(self):
        pass

