
# coding: utf-8

# # エンゼルの出現確率を予測する

import sqlite3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pymc as pm

import argparse

parser = argparse.ArgumentParser(description='argparser')
parser.add_argument('--filter', type=str, default='campaign is not 1')
parser.add_argument('--table', type=str, default='measurement')
parser.add_argument('--db', type=str, default='../data/choco-ball.db')
args = parser.parse_args()


def get_data(db_file='../data/choco-ball.db',
             table_name='measurement', filter_str=None):
    """
    dbファイルから計測データを取得する
    """
    con = sqlite3.connect(db_file)
    sql = 'SELECT '
    sql += 'measure_date,best_before,prd_number,weight,box_weight,ball_number,factory,shop,angel,campaign,taste '
    sql += ', (weight - box_weight), (weight - box_weight)/ball_number '
    sql += 'FROM ' + table_name + ' '
    if filter_str is not None:
        sql += 'WHERE ' + filter_str
    sql += ';'
    print(sql)
    sql_result = con.execute(sql)
    res = sql_result.fetchall()
    con.close()
    data = pd.DataFrame(
        res,
        columns=['measure_date', 'best_before',
                 'prd_number', 'weight', 'box_weight',
                 'ball_number', 'factory', 'shop',
                 'angel', 'campaign', 'taste',
                 'net_weight', 'mean_weight'])
    print('Shape of MeasurementData(record_num, n_columns) : {}'.format(data.shape))
    return data


def getMCMCResult(data, n_sample=15000, n_burn=5000):
    """
    MCMCでエンゼルの出現確率を予測する
    Args:
        data:エンゼルの観測結果(array)
        n_sample:MCMCシミュレーションの回数(integer)
        n_burn:捨てる数(integer)
    """
    # 出現確率pの事前分布
    p = pm.Uniform('p', lower=0, upper=1)
    # 観測を結びつける
    obs = pm.Bernoulli('obs', p, value=data, observed=True)

    # MCMC
    # Modelオブジェクト生成
    model = pm.Model([p, obs])
    mcmc = pm.MCMC(model)
    mcmc.sample(n_sample, n_burn)

    return mcmc.trace('p')[:]


def printAngelNumber(data):
    print('| 計測データ数 | {} |'.format(data.shape[0]))
    print('| 銀のエンゼル出現数 | {} |'.format((data['angel'] == 1).sum()))
    print('| 金のエンゼル出現数 | {} |'.format((data['angel'] == 2).sum()))
    return 0


def getAngelRate(data_angel, fig_name='fig/estimate_angel_rate_latest.png'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    p_trace = getMCMCResult(data_angel)
    ret = ax.hist(p_trace, bins=np.linspace(0, 0.5, 50), normed=True,
                  color="#0000FF", alpha=0.5, edgecolor="#0000FF", lw=2)
    ax.set_xlim([0, 0.5])
    N = len(p_trace)
    bci_g = np.sort(p_trace)[int(N * 0.95)]
    bci_l = np.sort(p_trace)[int(N * 0.05)]
    ax.vlines(x=bci_g, ymin=0, ymax=ret[0].max(),
              label='90% BayesCredibleInterval',
              color='red', linestyles='--', linewidths=2)
    ax.vlines(x=bci_l, ymin=0, ymax=ret[0].max(),
              color='red', linestyles='--', linewidths=2)
    ax.legend(loc="upper right")
    ax.set_title('observation number = %d' % (len(data_angel)))

    fig.savefig(fig_name)
    print '95% BayesCredibleInterval : {}-{}'.format(bci_l, bci_g)

    return 0


if __name__ == '__main__':
    #data = get_data(filter_str='taste=0')
    #data = get_data(filter_str='campaign is not 1')
    #data = get_data(filter_str='campaign is 1')
    data = get_data(filter_str=args.filter,
                    db_file=args.db, table_name=args.table)

    # 個数の集計
    printAngelNumber(data)

    # エンゼルの集計をする
    # 銀のエンゼル
    data_angel = np.array([1 if a == 1 else 0 for a in data['angel'].values])
    getAngelRate(data_angel=data_angel,
                 fig_name='fig/estimate_angel_rate_silver.png')
    # 金のエンゼル
    data_angel = np.array([1 if a == 2 else 0 for a in data['angel'].values])
    getAngelRate(data_angel=data_angel,
                 fig_name='fig/estimate_angel_rate_gold.png')
