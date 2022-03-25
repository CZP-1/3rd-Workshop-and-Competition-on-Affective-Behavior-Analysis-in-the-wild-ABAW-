import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def take_labels_ex(annot_txt_file):
    with open(annot_txt_file) as f:
        lines = f.readlines()
    labels = []
    for line in lines[1:]:
        labels.append(int(line))
    labels = np.array(labels)
    return labels


def take_labels_au(annot_txt_file):
    with open(annot_txt_file) as f:
        lines = f.readlines()
    labels = []
    for line in lines[1:]:
        labels.append([int(i) for i in line.split(',')])
        # import pdb; pdb.set_trace()
    labels = np.array(labels)
    return labels


def take_labels_va(annot_txt_file):
    with open(annot_txt_file) as f:
        lines = f.readlines()
    labels = []
    for line in lines[1:]:
        labels.append([float(i) for i in line.split(',')])
        # import pdb; pdb.set_trace()
    labels = np.array(labels)
    return labels


def take_images(data_file_name):
    images = glob.glob(data_file_name + '/*.*')
    images.sort()
    number_id = [os.path.split(image)[1].split('.')[0] for image in images]
    number_id = np.array(number_id, int)
    return images, number_id


def creat_df_video(annot_txt_file, data_file_name, path_save, type_partition):
    images_dir, number_id = take_images(data_file_name)
    df = pd.DataFrame(images_dir, columns=['image_id'])

    if type_partition == 'EXPR_Set':
        labels = take_labels_ex(annot_txt_file)
        df['labels_ex'] = labels[number_id - 1]
        df = df[df['labels_ex'].isin(range(8))].reset_index(drop=True)
        df.to_csv(path_save)
    elif type_partition == 'AU_Set':
        labels = take_labels_au(annot_txt_file)
        df['labels_au'] = [list(i) for i in labels[number_id - 1]]
        index_have_negative_value = []
        for index, i in enumerate(df['labels_au']):
            if -1 in i:
                index_have_negative_value.append(index)
        df = df.drop(index=index_have_negative_value).reset_index(drop=True)
        # df['labels_au'].apply(lambda x: str(x))
        df.to_csv(path_save)
    elif type_partition == 'VA_Set':
        labels = take_labels_va(annot_txt_file)
        df['labels_va'] = [list(i) for i in labels[number_id - 1]]
        index_have_negative_value = []
        for index, i in enumerate(df['labels_va']):
            if -5 in i:
                index_have_negative_value.append(index)
        df = df.drop(index=index_have_negative_value).reset_index(drop=True)
        df.to_csv(path_save)


def creat_df_test(path_save, df_test):
    df = pd.read_csv(df_test)
    for i in tqdm(df['folder_dir'], total=len(df)):
        images_dir, number_id = take_test_images(i)
        df1 = pd.DataFrame(images_dir, columns=['image_id'])
        df1['result'] = 0
        os.makedirs(path_save, exist_ok=True)
        df1.to_csv(path_save+os.path.split(i)[1] + '.csv')

def take_test_images(data_file_name):
    images = glob.glob(os.path.join('/ABAW2022/Test/cropped_aligned',data_file_name,'*.*'))
    images.sort()
    number_id = [os.path.split(image)[1].split('.')[0] for image in images]
    number_id = np.array(number_id, int)
    return images, number_id

def take_name_video(dir_path):
    return os.path.split(dir_path)[1].split('.')[0]


if __name__ == '__main__':
    # dir_images = '../data/origin_faces/'
    """
    dir_images = '/home/data2/xieguochen/datasets/ABAM/cropped_aligned/'
    dir_save_df_labels = ['./data_label/labels_save/expression/',
                          './data_label/labels_save/action_unit/']

    for set_data in ['Train_Set', 'Validation_Set']:
        # for i, type_partition in enumerate(['EXPR_Set', 'AU_Set']):
        for i, type_partition in enumerate(['EXPR_Set', 'AU_Set']):
            list_txt = glob.glob(f'/home/data4/CZP/data/Annotations/{type_partition}/{set_data}/*')

            os.makedirs(os.path.join(dir_save_df_labels[i], set_data), exist_ok=True)
            # import pdb; pdb.set_trace()

            for annotation in tqdm(list_txt, total=len(list_txt)):
                path_save = os.path.join(dir_save_df_labels[i], f'{set_data}_v1')
                os.makedirs(path_save, exist_ok=True)
                creat_df_video(annot_txt_file=annotation,
                               data_file_name=os.path.join(dir_images, take_name_video(annotation)),
                               path_save=os.path.join(path_save, take_name_video(annotation) + '.csv'),
                               type_partition=type_partition)
    """
    df_test = 'test_video.csv'
    data_save = './data_label/labels_save/expression/Test_Set/'
    creat_df_test(data_save,df_test)
    # dir_save_df_labels = ['../data/labels_save/action_unit/Test_Set/',
    #                       '../data/labels_save/valence_arousal/Test_Set/',
    #                       '../data/labels_save/expression/Test_Set/']
    #
    # df_test = ['./weight/au_test_len.csv',
    #            './weight/va_test_len.csv',
    #            './weight/ex_test_len.csv']
    # for i in range(3):
    #     creat_df_test(dir_save_df_labels[i], df_test=df_test[i])