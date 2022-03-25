import torch
from models.resnet import resnet50
import pickle
# model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # if weights_path:
        # import pickle
        # with open(weights_path, 'rb') as f:
        #     obj = f.read()
        # weights = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}
        # model.load_state_dict(weights)

# def prepare_model_relation(file_name='UnbiasedEmo_best'):
def prepare_model_relation(file_name='/home/data4/CZP/3rd_ABAW2021-master/weights/resnet50_ft_weight.pkl'):
    print('Model EmotionNet Loaded')
    with open(file_name,'rb') as fr:
        checkpoint = pickle.load(fr)
    # with open(file_name, 'rb') as f:
    #     obj = f.read()
    #     weights = {key: weight_dict for key, weight_dict in pickle.loads(obj, encoding='latin1').items()}
    #     checkpoint = torch.load_state_dict(weights)
        # model.load_state_dict(weights)
    # save_file = f'./weights/{file_name}.pkl'
    # checkpoint = torch.load(save_file)

    backbone = resnet50(pretrained=True)
    _model = backbone
    new_state_dict = {}
    # for key, values in checkpoint['state_dict'].items():
    for key, values in checkpoint.items():
        new_state_dict[key.replace('module.', '')] = values

    del new_state_dict['fc.weight']
    del new_state_dict['fc.bias']

    # backbone.load_state_dict(new_state_dict, strict=False)
    _model.load_state_dict(new_state_dict, strict=False)
    return _model
