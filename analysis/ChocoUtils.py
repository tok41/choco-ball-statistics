# coding: utf-8

# 目的
# - 共通関数群

import sqlite3
import pandas as pd


def get_data(db_file='../data/choco-ball.db',
             table_name='measurement', filter_str=None):
    """
    dbファイルから計測データを取得する

    TODO:
        エラー処理を入れる
    """
    con = sqlite3.connect(db_file)
    sql = 'SELECT '
    sql += 'measure_date, best_before, '
    sql += 'prd_number, weight, box_weight, '
    sql += 'ball_number, factory, shop, angel, campaign, taste, '
    sql += 'buyer, '
    sql += '(weight - box_weight), '
    sql += '(weight - box_weight)/ball_number '
    sql += 'FROM ' + table_name + ' '
    if filter_str is not None:
        sql += 'WHERE ' + filter_str
    sql += ';'
    print(sql)
    sql_result = con.execute(sql)
    res = sql_result.fetchall()
    con.close()
    data = pd.DataFrame(res, columns=['measure_date', 'best_before',
                                      'prd_number', 'weight', 'box_weight',
                                      'ball_number', 'factory', 'shop',
                                      'angel', 'campaign', 'taste',
                                      'buyer',
                                      'net_weight', 'mean_weight'])
    print('Shape of MeasurementData(record_num, n_columns) : {}'.format(
        data.shape))
    return data
