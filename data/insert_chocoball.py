# -*- coding: utf-8 -*-

import sys
import os
import sqlite3
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='argparser')
parser.add_argument('--file', type=str, default=None)
parser.add_argument('--table', type=str, default='measurement')
parser.add_argument('--db', type=str, default='choco-ball.db')
args = parser.parse_args()


def create_table(con, table_name):
    """
    データテーブルを作成
    """
    sql = 'CREATE TABLE ' + table_name + '('
    sql += 'id integer primary key,'
    sql += 'measure_date text,'
    sql += 'best_before text,'
    sql += 'prd_number text,'
    sql += 'weight real,'
    sql += 'box_weight real,'
    sql += 'ball_number integer,'
    sql += 'factory text,'
    sql += 'shop text,'
    sql += 'angel integer,'
    sql += 'campaign integer,'
    sql += 'taste integer, '
    sql += 'buyer text'
    sql += ');'
    print(sql)
    con.execute(sql)
    # create index
    sql = 'CREATE INDEX id_index on {}(id);'.format(table_name)
    con.execute(sql)


def insert_data(con, data_file, table_name):
    """
    CSVデータファイルを読み込み、全てのデータをinsertする
    """
    print('InsertInto : {} -> {}'.format(data_file, table_name))
    data = pd.read_csv(data_file, encoding="utf-8")
    con.executemany('insert into {} (measure_date,best_before,prd_number,weight,box_weight,ball_number,factory,shop,angel,campaign,taste, buyer) values (?,?,?,?,?,?,?,?,?,?,?,?)'.format(
        table_name), np.array(data))
    con.commit()


def main():
    print('insert data file : {}'.format(args.file))
    # open DB
    if os.path.exists(args.db):
        print('open DB file : {}'.format(args.db))
    else:
        print('create DB file : {}'.format(args.db))
    con = sqlite3.connect(args.db, isolation_level=None)

    # テーブルの存在確認
    sql = 'select count(*) from sqlite_master where type=\'table\' and name=\'{}\';'.format(args.table)
    sql_result = con.execute(sql)
    res = sql_result.fetchall()
    if res[0][0] < 0.5:
        print('Create Table : {}'.format(args.table))
        create_table(con, args.table)

    # データのinsert
    insert_data(con, args.file, args.table)

    con.close()
    return 0


if __name__ == '__main__':
    if args.file is None:
        print('Nothing Datafile : Use option "--file=FILENAME"')
        sys.exit(1)
    if os.path.exists(args.file):
        main()
    else:
        print('Not Exist Datafile : {}'.format(args.file))
        sys.exit(1)
