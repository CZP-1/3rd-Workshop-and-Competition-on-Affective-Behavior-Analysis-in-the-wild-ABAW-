import pandas as pd

def main():
    path_train = 'data_label/labels_save/multi_label/train_ex_all_v1.csv'
    path_val = 'data_label/labels_save/multi_label/valid_ex_all_v1.csv'
    data_train = pd.read_csv(path_train)
    data_val = pd.read_csv(path_val)
    data_all = pd.concat([data_train,data_val],ignore_index=True)
    # data_all = shuffle(data_all)
    data_all = data_all.reset_index(drop=True)
    path_save = 'data_label/labels_save/multi_label/train.csv'
    data_all.to_csv(path_save)

if __name__ == '__main__':
    main()