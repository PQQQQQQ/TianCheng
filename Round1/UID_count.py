import pandas as pd
import numpy as np
import datetime
import warnings

warnings.filterwarnings("ignore")


op_train = pd.read_csv('../operation_train_new.csv')
trans_train = pd.read_csv('../transaction_train_new.csv')

op_test = pd.read_csv('../operation_round1_new.csv')
trans_test = pd.read_csv('../transaction_round1_new.csv')

tag_train = pd.read_csv('../tag_train_new.csv')
sub = pd.read_csv('../sub.csv')


op_train_id = op_train.groupby('UID')
op_test_id = op_test.groupby('UID')
trans_train_id = trans_train.groupby('UID')
trans_test_id = trans_test.groupby('UID')


def feature_count(group):
    dct_cnt = {}
    dct_cnt['mode_o'] = group['mode'].unique().shape[0]
    dct_cnt['success_o'] = group['success'].unique().shape[0]
    dct_cnt['os_o'] = group['os'].unique().shape[0]
    dct_cnt['version_o'] = group['version'].unique().shape[0]    
    dct_cnt['device1_o'] = group['device1'].unique().shape[0]
    dct_cnt['device2_o'] = group['device2'].unique().shape[0]
    dct_cnt['device_code1_o'] = group['device_code1'].unique().shape[0]
    dct_cnt['device_code2_o'] = group['device_code2'].unique().shape[0]
    dct_cnt['device_code3_o'] = group['device_code3'].unique().shape[0]
    dct_cnt['mac1_o'] = group['mac1'].unique().shape[0]    
    dct_cnt['mac2_o'] = group['mac2'].unique().shape[0]
    dct_cnt['ip1_o'] = group['ip1'].unique().shape[0]
    dct_cnt['ip2_o'] = group['ip2'].unique().shape[0]
    dct_cnt['wifi_o'] = group['wifi'].unique().shape[0]    
    dct_cnt['geo_code_o'] = group['geo_code'].unique().shape[0]
    dct_cnt['ip1_sub_o'] = group['ip1_sub'].unique().shape[0]    
    dct_cnt['ip2_sub_o'] = group['ip2_sub'].unique().shape[0]
    dct_cnt['op_o'] = group.shape[0]
    dct_cnt = pd.Series(dct_cnt)
    return dct_cnt


cnt_op_train = op_train_id.apply(feature_count)
cnt_op_train.to_csv('cnt_op_train.csv')

cnt_op_test = op_test_id.apply(feature_count)
cnt_op_test.to_csv('cnt_op_test.csv')


def feature_count_t(group):
    dct_cnt = {}
    dct_cnt['channel_t'] = group['channel'].unique().shape[0]
    dct_cnt['trans_amt_t'] = group['trans_amt'].unique().shape[0]
    dct_cnt['amt_src1_t'] = group['amt_src1'].unique().shape[0]
    dct_cnt['merchant_t'] = group['merchant'].unique().shape[0]    
    dct_cnt['code1_t'] = group['code1'].unique().shape[0]
    dct_cnt['code2_t'] = group['code2'].unique().shape[0]
    dct_cnt['trans_type1_t'] = group['trans_type1'].unique().shape[0]
    dct_cnt['acc_id1_t'] = group['acc_id1'].unique().shape[0]
    dct_cnt['device_code1_t'] = group['device_code1'].unique().shape[0]
    dct_cnt['device_code2_t'] = group['device_code2'].unique().shape[0]
    dct_cnt['device_code3_t'] = group['device_code3'].unique().shape[0]
    dct_cnt['device1_t'] = group['device1'].unique().shape[0]
    dct_cnt['device2_t'] = group['device2'].unique().shape[0]
    dct_cnt['mac1_t'] = group['mac1'].unique().shape[0]    
    dct_cnt['ip1_t'] = group['ip1'].unique().shape[0]
    dct_cnt['bal_t'] = group['bal'].unique().shape[0]
    dct_cnt['amt_src2_t'] = group['amt_src2'].unique().shape[0]
    dct_cnt['acc_id2_t'] = group['acc_id2'].unique().shape[0]    
    dct_cnt['acc_id3_t'] = group['acc_id3'].unique().shape[0]
    dct_cnt['geo_code_t'] = group['geo_code'].unique().shape[0]    
    dct_cnt['trans_type2_t'] = group['trans_type2'].unique().shape[0]
    dct_cnt['market_code_t'] = group['market_code'].unique().shape[0]    
    dct_cnt['market_type_t'] = group['market_type'].unique().shape[0]
    dct_cnt['ip1_sub_t'] = group['ip1_sub'].unique().shape[0]
    dct_cnt['op_t'] = group.shape[0]
    dct_cnt = pd.Series(dct_cnt)
    return dct_cnt


cnt_trans_train = trans_train_id.apply(feature_count_t)
cnt_trans_train.to_csv('cnt_trans_train.csv')

cnt_trans_test = trans_test_id.apply(feature_count_t)
cnt_trans_test.to_csv('cnt_trans_test.csv')


def feature_count_op(group):
    dct_cnt = {}
    dct_cnt['mode_o_count'] = group['mode'].count()
    dct_cnt['success_o_count'] = group['success'].count()
    dct_cnt['os_o_count'] = group['os'].count()
    dct_cnt['version_o_count'] = group['version'].count()  
    dct_cnt['device1_o_count'] = group['device1'].count()
    dct_cnt['device2_o_count'] = group['device2'].count()
    dct_cnt['device_code1_o_count'] = group['device_code1'].count()
    dct_cnt['device_code2_o_count'] = group['device_code2'].count()
    dct_cnt['device_code3_o_count'] = group['device_code3'].count()
    dct_cnt['mac1_o_count'] = group['mac1'].count()    
    dct_cnt['mac2_o_count'] = group['mac2'].count()
    dct_cnt['ip1_o_count'] = group['ip1'].count()
    dct_cnt['ip2_o_count'] = group['ip2'].count()
    dct_cnt['wifi_o_count'] = group['wifi'].count()   
    dct_cnt['geo_code_o_count'] = group['geo_code'].count()
    dct_cnt['ip1_sub_o_count'] = group['ip1_sub'].count()    
    dct_cnt['ip2_sub_o_count'] = group['ip2_sub'].count()
    dct_cnt = pd.Series(dct_cnt)
    return dct_cnt


count_op_train = op_train_id.apply(feature_count_op)
count_op_train.to_csv('count_op_train.csv')

count_op_test = op_test_id.apply(feature_count_op)
count_op_test.to_csv('count_op_test.csv')


trans_train_columns = trans_train.columns.drop(['UID', 'day', 'time'])

def feature_count_trans(group):
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


count_trans_train = trans_train_id.apply(feature_count_trans)
count_trans_train.to_csv('count_trans_train.csv')

count_trans_test = trans_test_id.apply(feature_count_trans)
count_trans_test.to_csv('count_trans_test.csv')