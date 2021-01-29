
# coding: utf-8

# # エンゼルの出現確率を予測する

import numpy as np
import pandas as pd
import pymc3 as pm
from datetime import datetime as dt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ChocoUtils import get_data


def get_date_str():
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y-%m-%d')
    return tstr


def getAngelRate(data, n_sample=10000, n_chain=3, ax=None):
    # データの整理
    data_0 = data.query('campaign != 1')
    data_1 = data.query('campaign == 1')
    d = np.array([
        [sum(data_0['angel']==0), sum(data_0['angel']==1), sum(data_0['angel']==2)],
        [sum(data_1['angel']==0), sum(data_1['angel']==1), sum(data_1['angel']==2)]
    ])
    weight = np.array([[1.0, 1.0, 1.0],
                       [1.0, 0.0, 2.0]])
    # パラメータ推定
    with pm.Model() as model:
        alpha = [1., 1., 1.] # hyper-parameter of DirichletDist.
        pi = pm.Dirichlet('pi', a=np.array(alpha))
        for i in np.arange(d.shape[0]):
            piw = pi*weight[i]
            m = pm.Multinomial('m_%s'%(i), n=np.sum(d[i]), p=piw, observed=d[i])
        trace = pm.sample(n_sample, chains=n_chain)
    np.savetxt('trace_pi.csv', trace['pi'], delimiter=',')
    # Silver
    hpd_l, hpd_u = pm.hpd(trace['pi'][:,1])
    print('Silver : 95% HPD : {}-{}'.format(hpd_l, hpd_u))
    print('Silver ExpectedValue : {}'.format(trace['pi'][:,1].mean()))
    # Gold
    hpd_l, hpd_u = pm.hpd(trace['pi'][:,2])
    print('Gold : 95% HPD : {}-{}'.format(hpd_l, hpd_u))
    print('Gold ExpectedValue : {}'.format(trace['pi'][:,2].mean()))
    # save fig
    if ax is not None:
        pm.plot_posterior(trace['pi'][:,0], ax=ax[0])
        pm.plot_posterior(trace['pi'][:,1], ax=ax[1])
        pm.plot_posterior(trace['pi'][:,2], ax=ax[2])
        ax[0].set_title('Nothing')
        ax[1].set_title('SilverAngel')
        ax[2].set_title('GoldAngel')
    return trace


def printAngelNumber(data):
    print('| 計測データ数 | {} |'.format(data.shape[0]))
    print('| 銀のエンゼル出現数 | {} |'.format((data['angel'] == 1).sum()))
    print('| 金のエンゼル出現数 | {} |'.format((data['angel'] == 2).sum()))


def main(args):
    data = get_data(filter_str=args.filter,
                    db_file=args.db, table_name=args.table)
    printAngelNumber(data)
    fig = plt.figure(figsize=(12, 4))
    ax = fig.subplots(1,3)
    trace = getAngelRate(data, ax=ax)
    fig.savefig('fig/estimate_angel_rate_{}.png'.format(get_date_str()))
    return 0

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='argparser')
    parser.add_argument('--filter', type=str, default='taste is not 10')
    parser.add_argument('--table', type=str, default='measurement')
    parser.add_argument('--db', type=str, default='../data/choco-ball.db')
    args = parser.parse_args()

    ret = main(args)

