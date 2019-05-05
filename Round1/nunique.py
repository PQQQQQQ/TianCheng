# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 20:37:17 2018

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


#op_data = pd.concat([op_train, op_test], axis=0, ignore_index=True)
#
#op_train_length = op_train.shape[0]
#
#trans_data = pd.concat([trans_train, trans_test], axis=0, ignore_index=True)
#
#trans_train_length = trans_train.shape[0]
#
#
#UID_nuq = ['day', 'mode', 'success', 'time', 'os', 'version', 'device1',
#                 'device2', 'device_code1', 'device_code2', 'device_code3', 'mac1',
#                 'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']
#
#for feat in UID_nuq:
#    temp = op_data.groupby(feat)['UID'].nunique().reset_index().rename(columns={'UID': "%s_UID_nuq_num" % feat})
#    op_data = pd.merge(op_data, temp, how='left', on=[feat])
#
#op_train_1 = op_data[: op_train_length]
#op_train_1 = op_train_1.drop(UID_nuq, axis=1)
#op_train_1.to_csv('op_train_nunique.csv', index=False)
#
#op_test_1 = op_data[op_train_length :]
#op_test_1 = op_test_1.drop(UID_nuq, axis=1)
#op_test_1.to_csv('op_test_nunique.csv', index=False)
#
#
#UID_nuq_1 = ['channel', 'day', 'time', 'trans_amt', 'amt_src1', 'merchant', 'code1', 'code2',
#             'trans_type1', 'acc_id1', 'device_code1', 'device_code2', 'device_code3', 'device1',
#             'device2', 'mac1', 'ip1', 'bal', 'amt_src2', 'acc_id2', 'acc_id3',
#             'geo_code', 'trans_type2', 'market_code', 'market_type', 'ip1_sub']
#
#for feat in UID_nuq_1:
#    temp = trans_data.groupby(feat)['UID'].nunique().reset_index().rename(columns={'UID': "%s_UID_nuq_num_t" % feat})
#    trans_data = pd.merge(trans_data, temp, how='left', on=[feat])
#    
#    
#trans_train_1 = trans_data[: trans_train_length]
#trans_train_1 = trans_train_1.drop(UID_nuq_1, axis=1)
#trans_train_1.to_csv('trans_train_nunique.csv', index=False)
#
#trans_test_1 = trans_data[trans_train_length :]
#trans_test_1 = trans_test_1.drop(UID_nuq_1, axis=1)
#trans_test_1.to_csv('trans_test_nuniuqe.csv', index=False)    
    
    
def feature_nunique_op(group):
    dct_cnt = {}
    for feature in trans_train_columns:
        if group[feature].dtype == 'object':
            dct_cnt[feature+'_t_count'] = group[feature].count()
        else:
            dct_cnt[feature+'_t_count'] = group[feature].count()
            dct_cnt[feature+'_max'] = group[feature].max()
            dct_cnt[feature+'_min'] = group[feature].min()
            dct_cnt[feature+'_sum'] = group[feature].sum()
            dct_cnt[feature+'_mean'] = group[feature].mean()
            dct_cnt[feature+'_median'] = group[feature].median()
            dct_cnt[feature+'_std'] = group[feature].std()
            dct_cnt[feature+'_skw'] = group[feature].skew()
    
    dct_cnt = pd.Series(dct_cnt)
    return dct_cnt    
    
    
    
    
    
    
    
    
    