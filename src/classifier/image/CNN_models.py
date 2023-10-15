# from . import CNNModel
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
import torch



freeze_pretrain = {"mobilenet": [155, "classifier"],
                   "shufflenet": [167, "fc"],
                   "mnasnet": [155, "classifier"],
                   "resnet18": [59, "fc"],
                   "squeezenet": [49, "classifier"],
                   "resnet34": [107, "fc"],
                   }


class ModelBuilder:
    def __init__(self, pretrain=False):
        self.pretrain = pretrain
        # self.device = device

    def build(self, cls_num, backbone, device):
        self.device = device
        self.CNN = CNNModel(cls_num, backbone, load_pretrain=self.pretrain, device=self.device)
        if self.device != "cpu":
            self.CNN.model.cuda()
        self.params_to_update = self.CNN.model.parameters()
        self.softmax = torch.nn.Softmax(dim=1)
        return self.CNN.model

    def load_weight(self, path):
        self.CNN.load(path)

    def inference(self, img_tns):
        img_tensor_list = [torch.unsqueeze(img_tns, 0)]
        input_tensor = torch.cat(tuple(img_tensor_list), dim=0)
        self.image_batch_tensor = input_tensor.cuda() if self.device != "cpu" else input_tensor
        outputs = self.CNN.model(self.image_batch_tensor)
        outputs_tensor = outputs.data
        m_softmax = nn.Softmax(dim=1)
        outputs_tensor = m_softmax(outputs_tensor).to("cpu")
        return outputs_tensor







class LeNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(2704, 120)
        self.fc2 = nn.Linear(120, 10)
        #        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, num_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
        x = F.max_pool2d(F.relu(self.conv2(x)), (4, 4))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class CNNModel(object):
    def __init__(self, num_classes, model_name, load_pretrain=True, inp_size=224, device="cuda:0"):
        self.device = device
        if model_name == "inception":
            self.model = models.inception_v3()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            num_ftrs = self.model.AuxLogits.fc.in_features
            self.model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            # input_size = 299
        elif model_name == "resnet18":
            self.model = models.resnet18()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet34":
            self.model = models.resnet34()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet50":
            self.model = models.resnet50()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet101":
            self.model = models.resnet101()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "resnet152":
            self.model = models.resnet152()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "mobilenet":
            self.model = models.mobilenet_v2()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.model.last_channel, num_classes),
            )
        elif model_name == "shufflenet":
            self.model = models.shufflenet_v2_x1_0()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name, map_location=device))
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == "squeezenet":
            self.model = models.squeezenet1_1()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name), map_location=device)
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif model_name == "mnasnet":
            self.model = models.mnasnet1_0()
            if load_pretrain:
                self.model.load_state_dict(torch.load("weights/pretrain/%s.pth" % model_name), map_location=device)
            self.model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True),
                                                  nn.Linear(1280, num_classes))
        elif model_name == "vit":
            from .vit import ViT
            self.model = ViT(
                image_size=inp_size,
                patch_size=4,
                num_classes=num_classes,
                dim=512,                  # 512
                depth=6,
                heads=8,
                mlp_dim=512,
                dropout=0.1,
                emb_dropout=0.1
            )
        else:
            raise ValueError("Your pretrain model name is wrong!")

    def load(self, weight_path):
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))

