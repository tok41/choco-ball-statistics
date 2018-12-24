
# coding: utf-8

# # 目的
# - 基礎的な集計をする

import argparse
import ChocoUtils as util
from datetime import datetime as dt
import matplotlib.pyplot as plt
import sys
import os
import sqlite3
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='argparser')
parser.add_argument('--filter', type=str, default='taste=0')
parser.add_argument('--spec', type=float, default=28.0)
parser.add_argument('--table', type=str, default='measurement')
parser.add_argument('--db', type=str, default='../data/choco-ball.db')
parser.add_argument('--out', type=str, default='fig')
args = parser.parse_args()


def get_date_str():
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y-%m-%d')
    return tstr


# # 基礎集計
def output_hist(data, plt_file, step=0.1, spec=28.0):
    min_range = np.min([data['net_weight'].min(), spec]) * 0.9
    max_range = data['net_weight'].max() * 1.1
    b = np.arange(min_range, max_range, step)
    ret = plt.hist(data['net_weight'],
                   bins=b, color="#0000FF", alpha=0.5, edgecolor="#0000FF",
                   label='measure', density=True)
    plt.vlines(x=spec, ymin=0, ymax=ret[0].max(),
               colors='#FF0000', linewidths=2, label='spec')
    # 最尤推定パラメータの分布
    x = np.linspace(min_range, max_range, 300)
    y = stats.norm.pdf(
        x, loc=data['net_weight'].mean(), scale=data['net_weight'].std())
    plt.plot(x, y, lw=3, color='#0000FF', label='MLE')
    plt.legend()
    plt.xlabel('net weight [g]')
    plt.ylabel('frequency')
    plt.savefig(plt_file)
    print('save_figure : {}'.format(plt_file))


# メイン処理
def main():
    db_file = args.db
    table_name = args.table
    filter_str = args.filter
    output_dir = args.out
    # 計測データ取得
    m_data = util.get_data(db_file=db_file, table_name=table_name,
                           filter_str=filter_str)
    # ファイル名のラベルのために日付を取得
    t_str = get_date_str()
    # データ集計
    output_hist(data=m_data,
                plt_file='{}/base_hist_{}.png'.format(output_dir, t_str),
                spec=args.spec)
    # 集計結果表示用
    latest_date = m_data['measure_date'].max()
    latest_data = m_data[m_data['measure_date'] == latest_date][['measure_date', 'best_before',
                                                                 'weight', 'box_weight', 'ball_number', 'factory', 'shop',
                                                                 'angel', 'net_weight', 'mean_weight']]
    latest_data['angel'] = ['銀' if a ==
                            1 else 'なし' for a in latest_data['angel']]
    latest_data['net_weight'] = ["%2.3f" %
                                 (a) for a in latest_data['net_weight']]
    latest_data['mean_weight'] = ["%2.3f" %
                                  (a) for a in latest_data['mean_weight']]
    print(latest_data.to_csv(sep='|', index=False, header=False))
    # 基礎集計表示用
    print('| 計測データ数 | {} |'.format(m_data.shape[0]))
    print('| 銀のエンゼル出現数 | {} |'.format((m_data['angel'] == 1).sum()))
    print('| 金のエンゼル出現数 | {} |'.format((m_data['angel'] == 2).sum()))
    print('| 正味重量 | %2.3f | %2.3f | %2.3f | %2.3f |' % (
        (m_data['net_weight']).min(), (m_data['net_weight']).median(),
        (m_data['net_weight']).max(), (m_data['net_weight']).mean()))
    print('| 個数 | %2.3f | %2.3f | %2.3f | %2.3f |' % (
        (m_data['ball_number']).min(), (m_data['ball_number']).median(),
        (m_data['ball_number']).max(), (m_data['ball_number']).mean()))


if __name__ == '__main__':
    if os.path.exists(args.db):
        print('DB-File : {}'.format(args.db))
        main()
    else:
        print('Not Exist Datafile : {}'.format(args.db))
        sys.exit(1)
