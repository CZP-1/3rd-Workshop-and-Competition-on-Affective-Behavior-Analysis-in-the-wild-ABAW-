import pandas as pd
import glob
import os
import tqdm

def main():
    single_cat()
    # multi_cat()

def single_cat():
    # dir_save_df_labels = ['./data_label/labels_save/expression/',
    #                       './data_label/labels_save/action_unit/']
    
    # data = pd.DataFrame(data=None,columns=['image_id','labels_au'])
    # for __class in dir_save_df_labels:
    #     for set_data in ['Train_Set_v2', 'Validation_Set_v2']:
            
    #         list_txt = os.listdir(os.path.join(__class,set_data,"/*.csv"))
    #         for annotation in list_txt:
    #             data_single = pd.read_csv(annotation)
    #             data = pd.concat([data,data_single])
    #         data.to_csv(os.path.join(__class,set_data,'/.csv'))
    flag = 0       
    # data = []
    # list_txt = glob.glob("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/expression/Validation_Set_v1/*")
    list_txt = glob.glob("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/expression/Train_Set_v1/*")
    for annotation in list_txt:
        data_single = pd.read_csv(annotation,index_col=0)
        data_1 = pd.DataFrame(data_single)
        # data_1.drop(columns="Unnamed")
        if flag == 0:
            data = data_1
            flag = 1
        else:
            data = pd.concat([data,data_1],ignore_index=True)
            # result = data.append(data_1)
    # data.to_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/valid_ex_all_v1.csv")
    data.to_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/train_ex_all_v1.csv")

# def multi_cat():
#     data_ex_train = pd.read_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/train_ex_all_v2.csv",index_col=0)
#     data_ex_valid = pd.read_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/valid_ex_all_v2.csv",index_col=0)
#     data_au_train = pd.read_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/train_au_all_v2.csv",index_col=0)
#     data_au_valid = pd.read_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/valid_au_all_v2.csv",index_col=0)
    
#     data_ex_train = pd.DataFrame(data_ex_train)
#     data_ex_valid = pd.DataFrame(data_ex_valid)
#     data_au_train = pd.DataFrame(data_au_train)
#     data_au_valid = pd.DataFrame(data_au_valid)
    
#     data_train = pd.merge(data_ex_train,data_au_train,how='inner',on = 'image_id')
#     data_valid = pd.merge(data_ex_valid,data_au_valid,how='inner',on = 'image_id')
#     data_train.to_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/train_inner_ex_au_v2.csv")
#     data_valid.to_csv("/home/data4/CZP/3rd_ABAW2021-master/data_label/labels_save/multi_label/valid_inner_ex_au_v2.csv")

if __name__ == '__main__':
    main()
