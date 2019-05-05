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


var = ("hours","minutes","seconds")

op_train['second'] = op_train['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))
op_test['second'] = op_test['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))

trans_train['second'] = trans_train['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))
trans_test['second'] = trans_test['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))

op_train['second'] = op_train['day']*24*60*60 + op_train['second']
op_test['second'] = op_test['day']*24*60*60 + op_test['second']
trans_train['second'] = trans_train['day']*24*60*60 + trans_train['second']
trans_test['second'] = trans_test['day']*24*60*60 + trans_test['second']


def op_time_interval(group):
    time_diff = np.ediff1d(group['second'])
    op_id_interval = {}
    if len(time_diff) == 0:
        diff_mean = 0
        diff_std = 0
        diff_median = 0
        diff_zeros = 0
    else:
        diff_mean = np.mean(time_diff)
        diff_std = np.std(time_diff)
        diff_median = np.median(time_diff)
        diff_zeros = time_diff.shape[0] - np.count_nonzero(time_diff)
    op_id_interval['tmean_o'] = diff_mean
    op_id_interval['tstd_o'] = diff_std
    op_id_interval['tmedian_o'] = diff_median
    op_id_interval['tzeros_o'] = diff_zeros
    op_id_interval = pd.Series(op_id_interval)
    return op_id_interval

op_train_1 = op_train.sort_values(by='second')
op_train_1_id = op_train_1.groupby('UID')
op_train_inv = op_train_1_id.apply(op_time_interval)
op_train_inv.to_csv('op_train_inv.csv')

op_test_1 = op_test.sort_values(by='second')
op_test_1_id = op_test_1.groupby('UID')
op_test_inv = op_test_1_id.apply(op_time_interval)
op_test_inv.to_csv('op_test_inv.csv')


def trans_time_interval(group):
    time_diff = np.ediff1d(group['second'])
    op_id_interval = {}
    if len(time_diff) == 0:
        diff_mean = 0
        diff_std = 0
        diff_median = 0
        diff_zeros = 0
    else:
        diff_mean = np.mean(time_diff)
        diff_std = np.std(time_diff)
        diff_median = np.median(time_diff)
        diff_zeros = time_diff.shape[0] - np.count_nonzero(time_diff)
    op_id_interval['tmean_t'] = diff_mean
    op_id_interval['tstd_t'] = diff_std
    op_id_interval['tmedian_t'] = diff_median
    op_id_interval['tzeros_t'] = diff_zeros
    op_id_interval = pd.Series(op_id_interval)
    return op_id_interval


trans_train_1 = trans_train.sort_values(by='second')
trans_train_1_id = trans_train_1.groupby('UID')
trans_train_inv = trans_train_1_id.apply(trans_time_interval)
trans_train_inv.to_csv('trans_train_inv.csv')

trans_test_1 = trans_test.sort_values(by='second')
trans_test_1_id = trans_test_1.groupby('UID')
trans_test_inv = trans_test_1_id.apply(trans_time_interval)
trans_test_inv.to_csv('trans_test_inv.csv')