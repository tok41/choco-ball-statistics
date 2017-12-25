
# coding: utf-8

# # エンゼルの出現確率を予測する

import sys, os
import sqlite3
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anm

from datetime import datetime as dt

import pymc as pm
import scipy.stats as stats


def get_data(db_file='../data/choco-ball.db', table_name='measurement', filter_str=None):
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
    sql_result = con.execute(sql)
    res = sql_result.fetchall()
    con.close()
    data = pd.DataFrame(res, columns=['measure_date','best_before','prd_number','weight','box_weight','ball_number','factory','shop','angel','campaign','taste','net_weight','mean_weight'])
    print 'Shape of MeasurementData(record_num, n_columns) : {}'.format(data.shape)
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

if __name__ == '__main__':
    #data = get_data(filter_str='taste=0')
    #data = get_data(filter_str='campaign is not 1')
    data = get_data(filter_str='campaign is 1')
    data_angel = data['angel'].values
    
    fig = plt.figure(figsize = (10, 6))

    ### 静止画の保存
    p_trace = getMCMCResult(data_angel)
    ret = plt.hist(p_trace, bins=np.linspace(0, 0.5, 50), normed=True,
                       color="#0000FF", alpha=0.5, edgecolor="#0000FF", lw=2)
    plt.xlim([0, 0.5])
    N = len(p_trace)
    bci = np.sort(p_trace)[int(N*0.95)]
    plt.vlines(x=bci, ymin=0, ymax=ret[0].max(),
                   label='95% BayesCredibleInterval',
                   color='red', linestyles='--', linewidths=2)
    plt.legend(loc="upper right")
    plt.title('observation number = %d'%(len(data_angel)))
    
    plt.savefig('fig/estimate_angel_rate_latest.png')
    print '95% BayesCredibleInterval : {}'.format(bci)
    
