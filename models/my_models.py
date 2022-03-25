import torch
import torch.nn as nn
from models.efficientnet import efficientnet_b0,efficientnet_b1

from models.inception_resnetv1 import InceptionResnetV1
from models.resnest import resnest14d, resnest50d,resnest101e,resnest26d


class my_efficientnetb0(nn.Module):
    def __init__(self, num_classes_ex):
        super(my_efficientnetb0, self).__init__()
        
        self.backbone = efficientnet_b0()

        state_dict = torch.load('/home/data4/CZP/weights_pre/efficientnet_b0_ra-3dd342df.pth')
        self.backbone.load_state_dict(state_dict)

        self.backbone.fc = nn.Identity()

        self.fc1_1 = nn.Linear(in_features=1000, out_features=512)
        self.fc1_2 = nn.Linear(in_features=512, out_features=num_classes_ex)
        
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_b = self.backbone(x)
        out_1 = self.fc1_1(out_b)
        # out_ex = self.fc1_2(out_1)
        out_ex = self.fc1_2(self.dropout(self.relu(out_1)))

        return out_ex
class facenet(nn.Module):
    def __init__(self, num_classes_ex):
        super(facenet, self).__init__()
        
        self.backbone = InceptionResnetV1(pretrained='id')
        # self.backbone = InceptionResnetV1(pretrained='casia-webface')

        
        self.backbone.fc = nn.Identity()
        self.fc1_1 = nn.Linear(in_features=512, out_features=512)
        self.fc1_2 = nn.Linear(in_features=512, out_features=num_classes_ex)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out_b = self.backbone(x)
        out_1 = self.fc1_1(out_b)
        out_ex = self.fc1_2(self.dropout(self.relu(out_1)))
        return out_ex

class ResneSt50(nn.Module):
    def __init__(self, num_classes_ex,num_classes_au):
        super(ResneSt50, self).__init__()
        
        self.backbone = resnest50d(pretrained=True)

        # state_dict = torch.load('/home/data4/CZP/weights_pre/efficientnet_b0_ra-3dd342df.pth')
        # self.backbone.load_state_dict(state_dict)

        self.backbone.fc = nn.Identity()

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
        

        return out_ex,out_au

class ArcLoss4(nn.Module):
    def __init__(self, feature_num, class_num, s=10, m=0.5):
    	
        super().__init__()
        self.class_num = class_num
        self.feature_num = feature_num
        self.s = s
        self.m = torch.tensor(m)
        self.w = nn.Parameter(torch.rand(feature_num, class_num), requires_grad=True)  # 2*10

    def forward(self, feature):
        feature = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(self.w, dim=0)
        cos_theat = torch.matmul(feature, w) / 10
        sin_theat = torch.sqrt(1.0 - torch.pow(cos_theat, 2))
        cos_theat_m = cos_theat * torch.cos(self.m) - sin_theat * torch.sin(self.m)
        cos_theat_ = torch.exp(cos_theat * self.s)
        sum_cos_theat = torch.sum(torch.exp(cos_theat * self.s), dim=1, keepdim=True) - cos_theat_
        top = torch.exp(cos_theat_m * self.s)
        div = top / (top + sum_cos_theat)
        return div

# 实现方式2
class ArcLoss2(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=10):
        super().__init__()
        self.W = nn.Parameter(torch.randn(feature_dim, cls_dim), requires_grad=True)

    def forward(self, feature, m=1, s=10):
        x = nn.functional.normalize(feature, dim=1)
        w = nn.functional.normalize(self.W, dim=0)
        cos = torch.matmul(x, w)/10             # 求两个向量夹角的余弦值
        a = torch.acos(cos)                     # 反三角函数求得 α
        top = torch.exp(s*torch.cos(a+m))       # e^(s * cos(a + m))
        down2 = torch.sum(torch.exp(s*torch.cos(a)), dim=1, keepdim=True)-torch.exp(s*torch.cos(a))
        out = torch.log(top/(top+down2))
        return out