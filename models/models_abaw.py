from utils import prepare_model
import torch
import torch.nn as nn

from models import *
import pickle


class Multitask(nn.Module):
    def __init__(self, num_classes_ex,num_classes_au):
        super(Multitask, self).__init__()
        # self.backbone = prepare_model.prepare_model_relation()

        self.backbone = resnet.resnet50(pretrained=False)
        pretrained_vggface2 = './weight/resnet50_ft_weight.pkl'
        # pretrained_vggface2 = './weight/resnet50_scratch_weight.pkl'
        with open(pretrained_vggface2, 'rb') as f:
            pretrained_data = pickle.load(f)
        for k, v in pretrained_data.items():
            pretrained_data[k] = torch.tensor(v)

        self.backbone.fc = nn.Identity()
        self.backbone.load_state_dict(pretrained_data, strict=False)

        self.fc1_1 = nn.Linear(in_features=2048, out_features=512)
        self.fc1_2 = nn.Linear(in_features=512, out_features=num_classes_ex)
        self.fc2_1 = nn.Linear(in_features=2048, out_features=512)
        self.fc2_2 = nn.Linear(in_features=512, out_features=num_classes_au)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_b = self.backbone(x)

        out_ex = self.fc1_1(out_b)
        out_ex = self.fc1_2(self.dropout(self.relu(out_ex)))

        out_au = self.fc2_1(out_b)
        out_au = self.fc2_2(self.dropout(self.relu(out_au)))

        return out_ex, out_au
        # return out_ex
class r50_vgg(nn.Module):
    def __init__(self, num_classes):
        super(r50_vgg, self).__init__()
        # self.backbone = prepare_model.prepare_model_relation()

        self.backbone = resnet.resnet50(pretrained=False)
        pretrained_vggface2 = './weight/resnet50_ft_weight.pkl'
        # pretrained_vggface2 = './weight/resnet50_scratch_weight.pkl'
        with open(pretrained_vggface2, 'rb') as f:
            pretrained_data = pickle.load(f)
        for k, v in pretrained_data.items():
            pretrained_data[k] = torch.tensor(v)

        self.backbone.fc = nn.Identity()
        self.backbone.load_state_dict(pretrained_data, strict=False)

        self.fc1_1 = nn.Linear(in_features=2048, out_features=512)
        self.fc1_2 = nn.Linear(in_features=512, out_features=num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_b = self.backbone(x)
        out_ex = self.fc1_1(out_b)
        out_ex = self.fc1_2(self.dropout(self.relu(out_ex)))
        return out_ex
