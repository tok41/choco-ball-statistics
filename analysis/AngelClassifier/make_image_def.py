# -*- coding: utf-8 -*-

"""
ラベル付きの画像定義ファイルを作成する。
"""

# Imports
import os
import sqlite3
import numpy as np
import pandas as pd
import glob
#import tensorflow as tf


# tf.app.flags.DEFINE_string("img_path", "images", "image data path")
# tf.app.flags.DEFINE_string(
#     "train_list_path", "image_list_train.csv", "image data list file")
# tf.app.flags.DEFINE_string(
#     "test_list_path", "image_list_test.csv", "image data list file")
# tf.app.flags.DEFINE_string(
#     "db_path", "../../data/choco-ball.db", "DB file path")


def makeImageDefinition(img_path, db_file):
    """
    Args:
        img_path:画像ファイルのパス
        db_file:DB file path
    Returns:
        df_img_def:pd.DataFrame, [img_path, label]
    """
    # 画像ファイルのリスト作成
    files_exist = glob.glob(os.path.join(img_path, '*.png'))
    files = [os.path.basename(p) for p in files_exist]

    # DBのラベル付きファイルリストを取得
    con = sqlite3.connect(db_file)
    query = 'SELECT i.dataID, m.angel, i.path '
    query += 'FROM image  as i '
    query += 'LEFT JOIN measurement as m ON i.dataID = m.id '
    query += 'WHERE m.campaign is not 1 AND m.taste = 0;'
    sql_result = con.execute(query)
    res = sql_result.fetchall()
    con.close()
    data = pd.DataFrame(res, columns=['dataID', 'angel', 'base_path'])
    data['filename'] = [os.path.basename(p) for p in data['base_path']]

    # 実在するファイルと合わせる
    df_data = pd.merge(data, pd.DataFrame(
        {'filename': files, 'img_path': files_exist}),
        on='filename', how='inner')

    # 画像定義DF
    df_img_def = df_data[['img_path', 'angel']]

    return df_img_def


def splitTrainTest(df_img_def, rate=0.8, upsampling=False):
    """
    TrainデータとTestデータに分割
    Args:
        df_img_def : pd.DataFrame, [file_path, label]
        rate : train data rate, integer, default=0.8
    """
    def splitDF(data):
        N = data.shape[0]
        N_train = int(N*rate)
        N_test = N - N_train
        index = np.arange(N)
        np.random.shuffle(index)
        df_train = data.iloc[index[:N_train], :]
        df_test = data.iloc[index[N_train:], :]
        print('N={}, N_train={}, N_test={}'.format(N, N_train, N_test))
        return df_train, df_test
    posi_data = df_img_def[df_img_def['angel'] == 1]
    nega_data = df_img_def[df_img_def['angel'] == 0]
    posi_train, posi_test = splitDF(posi_data)
    nega_train, nega_test = splitDF(nega_data)
    if upsampling:
        print('simple upsampling positive_data : {} -> {}'.format(
            posi_train.shape[0], nega_train.shape[0]))
        posi_train = posi_train.sample(nega_train.shape[0], replace=True)
    df_train = pd.concat([posi_train, nega_train]).sample(
        frac=True).reset_index(drop=True)
    df_test = pd.concat([posi_test, nega_test]).sample(
        frac=True).reset_index(drop=True)
    return df_train, df_test


def main(argv):
    #FLAGS = tf.app.flags.FLAGS
    img_path = "images"
    db_path = "../../data/choco-ball.db"
    train_list_path = "image_list_train.csv"
    test_list_path = "image_list_test.csv"

    print("image_data_path : {}".format(img_path))

    # 画像定義ファイルの作成
    df_img = makeImageDefinition(img_path, db_path)
    print('image_num : {}'.format(df_img.shape[0]))
    df_train, df_test = splitTrainTest(df_img_def=df_img, rate=0.8)

    # 出力
    #df_img.to_csv(FLAGS.def_file_path, index=False, header=False)
    df_train.to_csv(train_list_path, index=False, header=False)
    df_test.to_csv(test_list_path, index=False, header=False)

    return 0


if __name__ == '__main__':
    main()
