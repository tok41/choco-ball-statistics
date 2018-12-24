
# coding: utf-8

# # エンゼルの出現確率を予測する

import argparse
import ChocoUtils as util
import pymc3 as pm
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')


parser = argparse.ArgumentParser(description='argparser')
parser.add_argument('--filter', type=str,
                    default='campaign is not 1 and taste is not 10')
parser.add_argument('--table', type=str, default='measurement')
parser.add_argument('--db', type=str, default='../data/choco-ball.db')
parser.add_argument('--out', type=str, default='fig')
args = parser.parse_args()


def getMCMCResult(data, n_sample=2000):
    """
    MCMCでエンゼルの出現確率を予測する
    Args:
        data:エンゼルの観測結果(array)
        n_sample:MCMCシミュレーションの回数(integer)
        n_burn:捨てる数(integer)
    """
    with pm.Model() as model_g:
        # theta = pm.Uniform('theta', lower=0, upper=1) # 一様分布
        theta = pm.Beta('theta', alpha=1.0, beta=1.0)
        obs = pm.Binomial('obs', n=len(data), p=theta, observed=sum(data))
        #obs = pm.Bernoulli('obs', p=theta, observed=data)
        start = pm.find_MAP()
        trace = pm.sample(n_sample, start=start, chains=4)
    return trace['theta']


def printAngelNumber(data):
    print('| 計測データ数 | {} |'.format(data.shape[0]))
    print('| 銀のエンゼル出現数 | {} |'.format((data['angel'] == 1).sum()))
    print('| 金のエンゼル出現数 | {} |'.format((data['angel'] == 2).sum()))
    return 0


def getAngelRate(data_angel, p_range=(0, 0.5), fig_name='fig/estimate_angel_rate_latest.png'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    p_trace = getMCMCResult(data_angel)
    ret = ax.hist(p_trace, bins=np.linspace(p_range[0], p_range[1], 50), density=True,
                  color="#0000FF", alpha=0.5, edgecolor="#0000FF", lw=2)
    ax.set_xlim([p_range[0], p_range[1]])
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
    print('95% BayesCredibleInterval : {}-{}'.format(bci_l, bci_g))
    print('ExpectedValue : {}'.format(p_trace.mean()))

    return 0


if __name__ == '__main__':
    data = util.get_data(filter_str=args.filter,
                         db_file=args.db, table_name=args.table)

    #
    output_dir = args.out
    t_str = util.get_date_str()

    # 個数の集計
    printAngelNumber(data)

    # エンゼルの集計をする
    # 銀のエンゼル
    data_angel = np.array([1 if a == 1 else 0 for a in data['angel'].values])
    getAngelRate(data_angel=data_angel, p_range=(0, 0.2),
                 fig_name='{}/estimate_angel_rate_silver_{}.png'.format(output_dir, t_str))
    # 金のエンゼル
    data_angel = np.array([1 if a == 2 else 0 for a in data['angel'].values])
    getAngelRate(data_angel=data_angel, p_range=(0, 0.05),
                 fig_name='{}/estimate_angel_rate_gold_{}.png'.format(output_dir, t_str))
