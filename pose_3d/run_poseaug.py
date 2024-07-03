from __future__ import print_function, absolute_import, division

import datetime
import os
import os.path as path
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def main(args):
    device = torch.device("cuda")
    print('==> Loading dataset...')
    data_dict = data_preparation(args)
