# -*- coding: utf-8 -*-

import sys
import os
import sqlite3
import argparse

parser = argparse.ArgumentParser(description='argparser')
parser.add_argument('--file', type=str, default=None)
parser.add_argument('--id', type=int, default=None)
parser.add_argument('--table', type=str, default='image')
parser.add_argument('--db', type=str, default='choco-ball.db')
args = parser.parse_args()


def create_table(con, table_name):
    """
    データテーブルを作成
    """
    sql = 'CREATE TABLE ' + table_name + '('
    sql += 'id integer primary key,'
    sql += 'dataID integer, '
    sql += 'path text'
    sql += ');'
    print(sql)
    con.execute(sql)


def insert_data(con, data_file, data_id, table_name):
    """
    data_file_pathとdataIDを指定して、テーブルに書き出す
    """
    print('InsertInto :{}:{} -> {}'.format(
        data_id, data_file, table_name))
    con.execute(
        'insert into {} (dataID, path) values ({},"{}")'.format(
            table_name, data_id, data_file))
    con.commit()


def main():
    print('insert package image : {}'.format(args.file))
    print('data ID : {}'.format(args.id))
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
    HOME = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(HOME, args.file)
    insert_data(con, full_path, args.id, args.table)

    con.close()
    return 0


if __name__ == '__main__':
    if args.file is None:
        print('Nothing Imagefile : Use option "--file=FILENAME"')
        sys.exit(1)
    if os.path.exists(args.file):
        main()
    else:
        print('Not Exist Datafile : {}'.format(args.file))
        sys.exit(1)
