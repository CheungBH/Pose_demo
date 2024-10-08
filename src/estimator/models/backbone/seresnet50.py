import torch
import torch.nn as nn
import torch.nn.functional as F
from ..SE.SE_module import SELayer
from ..utils.utils import parse_cfg, parse_se, parse_num_block, parse_channel


class SeBottleneck(nn.Module):
    """ SeBottleneck """
    expansion = 4

    def __init__(self, inplanes, outplanes, cfg, stride=1, downsample=None, reduction=0):
        super(SeBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], outplanes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        if reduction:
            self.se = SELayer(outplanes, reduction=reduction)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out


class SeResnet(nn.Module):
    """ SeResnet """

    def __init__(self, block, cfg=None):
        super(SeResnet, self).__init__()

        if cfg is None:
            # Construct config variable.
            # only purned layer
            self.inplanes = 64
            self.block = block
            self.planes = [256, 512, 1024, 2048]
            num_blocks = [3, 4, 6, 3]
            se_reduction = 16
            self.only_first_se = True
            cfg = [[64, 64]*3,
                   [128, 128]*4,
                   [256, 256]*6,
                   [512, 512]*3]
            cfg = [item for sub_list in cfg for item in sub_list]
        else:
            model_cfg = parse_cfg(cfg)
            se_reduction, self.only_first_se = parse_se(model_cfg)
            self.planes = model_cfg["residual"]
            cfg = parse_channel(model_cfg)
            self.inplanes = model_cfg["first_conv"]
            num_blocks = parse_num_block(model_cfg)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, self.planes[0], num_blocks[0], cfg=cfg[0:2*sum(num_blocks[0:1])], stride=1,
                                      reduction=se_reduction)
        self.layer2 = self.make_layer(block, self.planes[1], num_blocks[1],
                                  cfg = cfg[2*sum(num_blocks[0:1]):2*sum(num_blocks[0:2])], stride=2, reduction=se_reduction)
        self.layer3 = self.make_layer(block, self.planes[2], num_blocks[2],
                                  cfg = cfg[2*sum(num_blocks[0:2]):2*sum(num_blocks[0:3])], stride=2, reduction=se_reduction)
        self.layer4 = self.make_layer(block, self.planes[3], num_blocks[3],
                                  cfg = cfg[2*sum(num_blocks[0:3]):2*sum(num_blocks[0:4])], stride=2, reduction=se_reduction)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def make_layer(self, block, planes, blocks, cfg, stride=1, reduction=0):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        if downsample is not None:
            layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample, reduction=reduction))
        else:
            layers.append(block(self.inplanes, planes, cfg[0:2], stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            if not self.only_first_se:
                layers.append(block(self.inplanes, planes, cfg[2*i: 2*(i+1)], reduction=reduction))
            else:
                layers.append(block(self.inplanes, planes, cfg[2 * i: 2 * (i + 1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        # x = self.relu(self.bn0(self.conv0(x))) # 64 * h/2 * w/2
        x = self.layer1(x)  # 256 * h/2 * w/2
        x = self.layer2(x)  # 512 * h/4 * w/4
        x = self.layer3(x)  # 1024 * h/8 * w/8
        x = self.layer4(x)  # 2048 * h/16 * w/16

        # x = F.avg_pool2d(x, 2)
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        return x


def create(cfg=None): # with maxpooling
    return SeResnet(SeBottleneck, cfg)


def test():
    net = create(cfg='../util/pruned/cfg_shortcut_seresnet50.json')
    y = net(torch.randn(1, 3, 64, 64))
    print(net, file=open("SEResnet.txt","w"))
    print(y.size())


if __name__ == '__main__':
    test()
