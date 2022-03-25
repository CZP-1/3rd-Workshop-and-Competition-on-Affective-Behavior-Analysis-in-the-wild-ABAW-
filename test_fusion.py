import argparse
import warnings

from torch.utils.data import DataLoader

from data_loader import *
from models.models_abaw import *
from utils import *

from models.my_models import *
from models.inception_v4 import inception_v4
from models.resnest import resnest14d, resnest50d,resnest101e,resnest26d 

from models.efficientnet import efficientnet_b0,efficientnet_b1, efficientnet_b2

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='MuSE Training')

parser.add_argument('-b', '--batch_size', default=512, type=int, help='batch size')

args = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

def count_frames(name_video):
    df_single = pd.read_csv('data_label/labels_save/expression/Test_Set/' + name_video + '.csv')
    len_video = total=len(df_single)
    start_frame = df_single.image_id[0]
    over_frame = df_single.image_id[len_video-1]
    start_frame_num = int(start_frame.split('.')[0].split('/')[-1])
    over_frame_num = int(over_frame.split('.')[0].split('/')[-1])
    start = start_frame_num
    over = over_frame_num
    count = over_frame_num - start_frame_num +1
    
    if(over != over_frame_num):
        print(name_video)

    return start,over,count

def main():
    df_video = pd.read_csv('test_video.csv')

    model1 = facenet(num_classes_ex=8)
    checkpoint1 = torch.load('weights_kfold/InceptionResnetV1_all.pth')
    model1.load_state_dict(checkpoint1)

    model2 = r50_vgg(num_classes=8)
    checkpoint2 = torch.load('weights_kfold/r50_vgg_all.pth')
    model2.load_state_dict(checkpoint2)

    model3 = efficientnet_b0(num_classes=8,pretrained=True)
    # model3 = my_efficientnetb0(num_classes_ex=8)
    checkpoint3 = torch.load('weights_kfold/efficientnet_b0_all.pth')
    model3.load_state_dict(checkpoint3)

    model1.to(device)
    model2.to(device)
    model3.to(device)


    model1.eval()
    model2.eval()
    model3.eval()

    for ind, name_video in tqdm(enumerate(df_video['folder_dir']), total=len(df_video)):
        
        with torch.no_grad():
            cat_preds = []
            df = pd.read_csv('data_label/labels_save/expression/Test_Set/' + name_video + '.csv')
            start, over, count= count_frames(name_video)
            test_dataset = Aff2_Dataset_static_multitask_test(df=df,
                                                          transform=test_transform)
            test_loader_ex = DataLoader(dataset=test_dataset,
                                    batch_size=args.batch_size,
                                    num_workers=4,
                                    shuffle=True,
                                    drop_last=False)
            for batch_idx, samples in tqdm(enumerate(test_loader_ex), total=len(test_loader_ex),
                                           desc='Test_mode'):
                images = samples['images'].to(device).float()
                pred_cat1 = model1(images)
                pred_cat2 = model2(images)
                pred_cat3 = model3(images)

                pred_cat1 = F.softmax(pred_cat1)
                pred_cat2 = F.softmax(pred_cat2)
                pred_cat3 = F.softmax(pred_cat3)

                x1 = 1
                x2 = 1
                x3 = 1
                pred_cat = x1*pred_cat1 + x2*pred_cat2 + x3*pred_cat3
                pred_cat = torch.argmax(pred_cat, dim=1)

                cat_preds.append(pred_cat.detach().cpu().numpy())

            cat_preds = np.concatenate(cat_preds, axis=0)
        # if name_video == '134-30-1280x720':
        #     print('1')
        df['result'] = cat_preds
        df['image_id'] = df.image_id.apply(lambda x: int(os.path.split(x)[1].split('.')[0]))
        
        df1 = pd.DataFrame(columns=['image_id'],
                           data=list(set(range(start, over+1)).difference(set(df.image_id))))
        df1['result'] = -1
        # print(name_video)
        df2 = pd.concat([df, df1])
        df2 = df2.sort_values(by=['image_id']).reset_index(drop=True)
        path = 'results/expression/test1_all/'
        os.makedirs(path, exist_ok=True)

        file = open(path + name_video + '.txt', "w")
        file.write("Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other, \n")
        file.close()
        # import pdb; pdb.set_trace()
        df2 = df2.drop(columns='image_id')
        df2 = df2.result
        df2.to_csv(path + name_video + '.txt', header=None, index=None, sep=',', mode='a')


        # print(len((df2)))
        # break
if __name__ == '__main__':
    main()
    
