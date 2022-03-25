import argparse
import warnings
from models.my_models import ArcLoss2, ArcLoss4

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import trange
from sklearn.utils import shuffle

from data_loader import *
from metrics import *
from utils import *
from models.efficientnet import efficientnet_b0
from models.python.emotionnet import EmotionNet
from models.models_abaw import *
from models.inception_v4 import inception_v4
from models.resnest import resnest14d, resnest50d,resnest101e,resnest26d 
from models.my_models import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--num_epochs', default=30, type=int, help='number epochs')
parser.add_argument('--weight_decay', type=float, default=5e-4)

args = parser.parse_args()

device = "cuda:2" if torch.cuda.is_available() else 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = 1

def get_kfold_data(k,i):
    path_train = 'data_label/labels_save/multi_label/train_ex_all_v1.csv'
    path_val = 'data_label/labels_save/multi_label/valid_ex_all_v1.csv'
    data_train = pd.read_csv(path_train)
    data_val = pd.read_csv(path_val)
    data_all = pd.concat([data_train,data_val],ignore_index=True)
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
    # if i == k-1 :
    #     data_kfold_val = data_all[i*one_kold_data:]
    # else:
    #     data_kfold_val = data_all[i*one_kold_data:(i+1)*one_kold_data]
    data_kfold_train = pd.concat([data_all,data_kfold_val], ignore_index=True, verify_integrity=True)
    data_kfold_train.drop_duplicates(subset=['image_id'],keep=False,inplace=True)
    data_kfold_train = data_kfold_train.reset_index(drop=True)
    data_kfold_val = data_kfold_val.reset_index(drop=True)

    return data_kfold_train, data_kfold_val

def main():
    seed_everything()
    
    k=5
    for i in range(k):
        train_data,val_data=get_kfold_data(k,i)
        transform_train = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=(112, 112)),
                                        #   transforms.Resize(size=(224, 224)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                        # transforms.RandomErasing(),
                                        ])

        transform_val = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize(size=(112, 112)),
                                        #  transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                        ])

        #AFF-Wild2
        train_dataset_ex = Aff2_Dataset_static_multitask(
            df=train_data,
            transform=transform_train, type_partition='ex')
        train_loader_ex = DataLoader(dataset=train_dataset_ex,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    shuffle=True,
                                    #  sampler=samplers,
                                    drop_last=False,
                                    )

        valid_dataset_ex = Aff2_Dataset_static_multitask(
            df=val_data,
            transform=transform_val, type_partition='ex')

        valid_loader_ex = DataLoader(dataset=valid_dataset_ex,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    shuffle=False,
                                    drop_last=False,
                                    )



        model = facenet(num_classes_ex=8)
        # model = r50_vgg(num_classes=8)
        # model =  efficientnet_b2(num_classes=8,pretrained=True)
        # model = efficientnet_b0(num_classes=512,pretrained=True)
        print(model)

        model.to(device)
        criterion1 = FocalLoss_Ori(num_class=8, gamma=2.0, ignore_index=255, reduction='mean')
        criterion2 = nn.BCEWithLogitsLoss()
        criterion = nn.CrossEntropyLoss()
        criterion4 = ArcLoss4(512,8)
        
        criterion4.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler_steplr = CosineAnnealingLR(optimizer, args.num_epochs, eta_min=1e-4, last_epoch=-1)

        optimizer.zero_grad()
        optimizer.step()
        best_scores_ex = 0
        
        with trange(args.num_epochs, total=args.num_epochs, desc='Epoch') as t:
            for epoch in t:
                t.set_description('Epoch %i' % epoch)
                cost_list = 0
                scheduler_steplr.step(epoch)
                model.train()
                
                for batch_idx, samples in tqdm(enumerate(train_loader_ex), total=len(train_loader_ex)):
                    optimizer.zero_grad()
                    images = samples['images'].to(device).float()
                    labels_cat = samples['labels'].to(device).long()
                    pred_cat = model(images)
                    # loss = criterion4(pred_cat)
                    # loss = criterion(loss, labels_cat)
                    loss = criterion1(pred_cat, labels_cat) 

                    cost_list += loss.item()
                    

                    loss.backward()
                    optimizer.step()

                    t.set_postfix(Loss=f'{cost_list / (batch_idx + 1):04f}',
                                Batch=f'{batch_idx + 1:03d}/{len(train_loader_ex):03d}',
                                Lr=optimizer.param_groups[0]['lr'])

                model.eval()
                with torch.no_grad():
                    cat_preds = []
                    cat_labels = []
                    for batch_idx, samples in tqdm(enumerate(valid_loader_ex), total=len(valid_loader_ex),
                                                desc='Valid_mode'):
                        images = samples['images'].to(device).float()
                        labels_cat = samples['labels'].to(device).long()

                        # pred_cat, _ = model(images)
                        pred_cat = model(images)

                        
                        # pred_cat = criterion4(pred_cat)
                        pred_cat = F.softmax(pred_cat)
                        pred_cat = torch.argmax(pred_cat, dim=1)
                        cat_preds.append(pred_cat.detach().cpu().numpy())
                        cat_labels.append(labels_cat.detach().cpu().numpy())

                    cat_preds = np.concatenate(cat_preds, axis=0)
                    cat_labels = np.concatenate(cat_labels, axis=0)

                    f1, acc = EXPR_metric(cat_preds, cat_labels)
                    print(f'f1_ex = {f1} \n'
                        f'acc_ex = {acc} \n')
                    if best_scores_ex < f1:
                        best_scores_ex = f1
                        # os.makedirs('./weight', exist_ok=True)
                        path_save = 'weights_kfold/'+'facenet'+ str(i+1)+'_fold.pth'
                        torch.save(model.state_dict(), path_save)
                        # torch.save(model, f'./weights/efficientb0_arcface.pth')
            print('fold: ',i)
            print(best_scores_ex)


if __name__ == '__main__':
    main()
