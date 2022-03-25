import argparse
import warnings

from torch.utils.data import DataLoader

from data_loader import *
from models.models_abaw import *
from utils import *
from metrics import *

warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = 1

parser = argparse.ArgumentParser(description='MuSE Training')

parser.add_argument('-b', '--batch_size', default=512, type=int, help='batch size')
parser.add_argument('-p', '--partition', default='ex', type=str, help='partition')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'


def main():
    
    transform_val = transforms.Compose([transforms.ToPILImage(),
                                    #  transforms.Resize(size=(112, 112)),
                                     transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                     ])
    valid_dataset_ex = Aff2_Dataset_static_multitask(
        df=pd.read_csv('./data_label/labels_save/multi_label/valid_ex_all_v1.csv'),
        # df=pd.read_csv('/home/data4/CZP/datasets/Affectnet/val_set_all.csv'),
        transform=transform_val, type_partition='ex')


    valid_loader_ex = DataLoader(dataset=valid_dataset_ex,
                                 batch_size=args.batch_size,
                                 num_workers=4,
                                 shuffle=False,
                                 drop_last=False,
                                 )

    model = Resnet_Multitask()
    # checkpoint = torch.load('/home/data4/CZP/weights_pre/swin_base_patch4_window7_224.pth')
    print(model)
    
    model.to(device)

    best_scores_ex = 0
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
                    pred_cat = F.softmax(pred_cat)
                    pred_cat = torch.argmax(pred_cat, dim=1)

                    cat_preds.append(pred_cat.detach().cpu().numpy())
                    cat_labels.append(labels_cat.detach().cpu().numpy())

        cat_preds = np.concatenate(cat_preds, axis=0)
        cat_labels = np.concatenate(cat_labels, axis=0)

        f1, acc = EXPR_metric(cat_preds, cat_labels)
        if best_scores_ex < f1:
            best_scores_ex = f1
        print(f'f1_ex = {f1} \n'
                      f'acc_ex = {acc} \n')
    print(best_scores_ex)                  




if __name__ == '__main__':
    main()

