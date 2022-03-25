# This file is part of EmotionNet2 a system for predicting facial emotions
# Copyright (C) 2018  Maeve Kennedy
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>

from fileinput import filename
import torchvision.models.resnet as resnet
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import shutil
# For windowing
import numpy as np
import matplotlib.pyplot as plt

# For one file pass
import tempfile
# import face_detector
import os.path
from PIL import Image
# import extract_faces
# from models.python

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)


class EmotionNet(nn.Module):
    def __init__(self,num_classes_ex):
        super(EmotionNet, self).__init__()
        block = resnet.BasicBlock
        # num_classes = 7
        self.model = resnet.ResNet(block, [3, 4, 23, 3], num_classes_ex)
        # if torch.cuda.is_available():
        #     self.model.cuda()
        # self.bestaccur = 0.0s
        
        filename = 'models/checkpoint.pth.tar'
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(filename))
        else:
            self.model.load_state_dict(torch.load(filename, lambda storage, loc: storage))
        self.model.fc = nn.Identity()
        # self.fc1_1 = nn.Linear(in_features=2048, out_features=512)
        self.fc1_1 = nn.Linear(in_features=512, out_features=512)
        self.fc1_2 = nn.Linear(in_features=512, out_features=num_classes_ex)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_b = self.model(x)

        out_ex = self.fc1_1(out_b)
        out_ex = self.fc1_2(self.dropout(self.relu(out_b)))

        return out_ex