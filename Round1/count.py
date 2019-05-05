# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:00:51 2018

@author: PQ
"""

import pandas as pd
import numpy as np
import datetime
import warnings

warnings.filterwarnings("ignore")


op_train = pd.read_csv('../data/operation_train_new.csv')
trans_train = pd.read_csv('../data/transaction_train_new.csv')

op_test = pd.read_csv('../data/operation_round1_new.csv')
trans_test = pd.read_csv('../data/transaction_round1_new.csv')

tag_train = pd.read_csv('../data/tag_train_new.csv')
sub = pd.read_csv('../data/sub.csv')


op_data = pd.concat([op_train, op_test], axis=0, ignore_index=True)

op_train_length = op_train.shape[0]

trans_data = pd.concat([trans_train, trans_test], axis=0, ignore_index=True)

trans_train_length = trans_train.shape[0]


feature = ['day', 'mode', 'success', 'time', 'os', 'version', 'device1',
           'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1',
           'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']

for feat in feature:
    op_n = op_data[feat].value_counts().reset_index()
    op_n.columns = [feat, feat+'_n_count']
    op_data = pd.merge(op_data, op_n, how='left', on=[feat])
 
    
op_train_1 = op_data[: op_train_length]
op_train_1 = op_train_1.drop(feature, axis=1)
op_train_1.to_csv('n_count_op_train.csv', index=False)

op_test_1 = op_data[op_train_length :]
op_test_1 = op_test_1.drop(feature, axis=1)
op_test_1.to_csv('n_count_op_test.csv', index=False)
    

feature_1 = ['channel', 'day', 'time', 'trans_amt', 'amt_src1', 'merchant', 'code1', 'code2',
             'trans_type1', 'acc_id1', 'device_code1', 'device_code2', 'device_code3', 'device1',
             'device2', 'mac1', 'ip1', 'bal', 'amt_src2', 'acc_id2', 'acc_id3',
             'geo_code', 'trans_type2', 'market_code', 'market_type', 'ip1_sub']

for feat in feature_1:
    trans_n = trans_data[feat].value_counts().reset_index()
    trans_n.columns = [feat, feat+'_nt_count']
    trans_data = pd.merge(trans_data, trans_n, how='left', on=[feat])
    
    
trans_train_1 = trans_data[: trans_train_length]
trans_train_1 = trans_train_1.drop(feature_1, axis=1)
trans_train_1.to_csv('n_count_trans_train.csv', index=False)

trans_test_1 = trans_data[trans_train_length :]
trans_test_1 = trans_test_1.drop(feature_1, axis=1)
trans_test_1.to_csv('n_count_trans_test.csv', index=False)