import argparse
from subprocess import check_output
import warnings

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import trange
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from models.my_models import *
from data_loader import *
from metrics import *
from utils import *
from models.efficientnet import efficientnet_b0
from models.python.emotionnet import EmotionNet
from models.models_abaw import *
from models.inception_v4 import inception_v4
from models.resnest import resnest14d, resnest50d,resnest101e,resnest26d 
# from models.my_models import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=30, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:3" if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = 1

def get_kfold_data(k,i):
    path_train = 'data_label/labels_save/multi_label/train_ex_all_v1.csv'
    path_val = 'data_label/labels_save/multi_label/valid_ex_all_v1.csv'
    data_train = pd.read_csv(path_train)
    data_val = pd.read_csv(path_val)
    data_all = pd.concat([data_train,data_val],ignore_index=True)
    # data_all = shuffle(data_all)
    data_all = data_all.reset_index(drop=True)
    len_data = len(data_all)
    one_kold_data = int(len_data/k)
    if i==0:
        data_kfold_val = data_all[0:176240]
    elif i==1:
        data_kfold_val = data_all[176240:348072]
    elif i==2:
        data_kfold_val = data_all[348072:520083]
    elif i==3:
        data_kfold_val = data_all[520083:692339]
    elif i==4:
        data_kfold_val = data_all[692339:]
    data_kfold_train = pd.concat([data_all,data_kfold_val], ignore_index=True, verify_integrity=True)
    data_kfold_train.drop_duplicates(subset=['image_id'],keep=False,inplace=True)
    data_kfold_train = data_kfold_train.reset_index(drop=True)
    data_kfold_val = data_kfold_val.reset_index(drop=True)

    return data_kfold_train, data_kfold_val

def main():
    seed_everything()
    
    k=5
    for i in range(k):
        i=4
        train_data,val_data=get_kfold_data(k,i)

        transform_val = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=(112, 112)),
                                        #  transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])

        valid_dataset_ex = Aff2_Dataset_static_multitask(
            df=val_data,
            transform=transform_val, type_partition='ex')

        valid_loader_ex = DataLoader(dataset=valid_dataset_ex,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    shuffle=False,
                                    drop_last=False,
                                    )

        model1 = facenet(num_classes_ex=8)
        checkpoint1 = torch.load('./weights_kfold/InceptionResnetV1_kfold_'+str(i+1)+'_fold.pth')
        # checkpoint1 = torch.load('weights/InceptionResnetV1_vgg.pth')
        # model1 = torch.load('weights/InceptionResnetV1_vgg.pth')
        model1.load_state_dict(checkpoint1)

        model2 = r50_vgg(num_classes=8)
        checkpoint2 = torch.load('./weights_kfold/r50_vgg_kfold_new_'+str(i+1)+'_fold.pth')
        # checkpoint2 = torch.load('weights/multitask_resnet50_vgg_ex.pth')
        # model2 = torch.load('weight/resnet50.pth')
        model2.load_state_dict(checkpoint2)

        model3 = efficientnet_b0(num_classes=8,pretrained=True)
        # model3 = my_efficientnetb0(num_classes_ex=8)
        checkpoint3 = torch.load('./weights_kfold/efficientnet_b0_kfold_'+str(i+1)+'_fold.pth')
        # model3 = torch.load('weights/efficientet_ori.pth')
        # checkpoint3 = torch.load('weights/efficientet_ori.pth')
        model3.load_state_dict(checkpoint3)


        # print(model)

        model1.to(device)
        model2.to(device)
        model3.to(device)


        model1.eval()
        model2.eval()
        model3.eval()


        with torch.no_grad():
            cat_preds = []
            cat_preds1 = []
            cat_preds2 = []
            cat_preds3 = []
            cat_labels = []
            for batch_idx, samples in tqdm(enumerate(valid_loader_ex), total=len(valid_loader_ex),
                                        desc='Valid_mode'):
                images = samples['images'].to(device).float()
                labels_cat = samples['labels'].to(device).long()

                pred_cat1 = model1(images)
                pred_cat2 = model2(images)
                pred_cat3 = model3(images)

                pred_cat1 = F.softmax(pred_cat1)
                pred_cat2 = F.softmax(pred_cat2)
                pred_cat3 = F.softmax(pred_cat3)

                x1 = 0.4
                x2 = 0
                x3 = 0.6
                pred_cat = x1*pred_cat1 + x2*pred_cat2 + x3*pred_cat3
                pred_cat = torch.argmax(pred_cat, dim=1)
                
                pred_cat1 = torch.argmax(pred_cat1, dim=1)
                pred_cat2 = torch.argmax(pred_cat2, dim=1)
                pred_cat3 = torch.argmax(pred_cat3, dim=1)

                cat_preds.append(pred_cat.detach().cpu().numpy())
                cat_preds1.append(pred_cat1.detach().cpu().numpy())
                cat_preds2.append(pred_cat2.detach().cpu().numpy())
                cat_preds3.append(pred_cat3.detach().cpu().numpy())

                cat_labels.append(labels_cat.detach().cpu().numpy())

            cat_preds = np.concatenate(cat_preds, axis=0)
            cat_preds1 = np.concatenate(cat_preds1, axis=0)
            cat_preds2 = np.concatenate(cat_preds2, axis=0)
            cat_preds3 = np.concatenate(cat_preds3, axis=0)

            cat_labels = np.concatenate(cat_labels, axis=0)

            f1_fusion = f1_score(cat_labels, cat_preds, average='macro')
            f1_1 = f1_score(cat_labels, cat_preds1, average='macro')
            f1_2 = f1_score(cat_labels, cat_preds2, average='macro')
            f1_3 = f1_score(cat_labels, cat_preds3, average='macro')
            
            acc_fusion = accuracy_score(cat_labels, cat_preds)
            acc_1 = accuracy_score(cat_labels, cat_preds1)
            acc_2 = accuracy_score(cat_labels, cat_preds2)
            acc_3 = accuracy_score(cat_labels, cat_preds3)
            
            print(f'f1_fusion = {f1_fusion} \t'
                    f'acc_fusion = {acc_fusion} \n'
                    f'f1_1 = {f1_1} \t'
                    f'acc_1 = {acc_1} \n'
                    f'f1_2 = {f1_2} \t'
                    f'acc_2 = {acc_2} \n'
                    f'f1_3 = {f1_3} \t'
                    f'acc_3 = {acc_3} \n')
            return

if __name__ == '__main__':
    # k=5
    # for i in range(5):
    #     a,b = get_kfold_data(k,i)
    main()
