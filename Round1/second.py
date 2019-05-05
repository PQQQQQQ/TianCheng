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


var = ("hours","minutes","seconds")

op_train['second'] = op_train['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))
op_test['second'] = op_test['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))

trans_train['second'] = trans_train['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))
trans_test['second'] = trans_test['time'].apply(lambda x:int(datetime.timedelta(**{k:int(v) for k,v in zip(var,x.strip().split(":"))}).total_seconds()))

op_train['second'] = op_train['day']*24*60*60 + op_train['second']
op_test['second'] = op_test['day']*24*60*60 + op_test['second']
trans_train['second'] = trans_train['day']*24*60*60 + trans_train['second']
trans_test['second'] = trans_test['day']*24*60*60 + trans_test['second']


feature = ['mode', 'success', 'os', 'version', 'device1', 'device2', 'device_code1', 'device_code2',
       'device_code3', 'mac1', 'mac2', 'ip1', 'ip2', 'wifi', 'geo_code', 'ip1_sub', 'ip2_sub']


for feat in feature:
    temp = op_train.groupby(['UID',feat])['second'].mean().reset_index().rename(columns={'second': feat+'_o_second'})
    op_train = pd.merge(op_train, temp, on=['UID',feat], how='left')

op_train_1 = op_train.drop(feature, axis=1)
op_train_1 = op_train_1.drop(['day', 'time', 'second'], axis=1)
op_train_1.to_csv('op_train_second.csv', index=False)

for feat in feature:
    temp = op_test.groupby(['UID',feat])['second'].mean().reset_index().rename(columns={'second': feat+'_o_second'})
    op_test = pd.merge(op_test, temp, on=['UID',feat], how='left')

op_test_1 = op_test.drop(feature, axis=1)
op_test_1 = op_test_1.drop(['day', 'time', 'second'], axis=1)
op_test_1.to_csv('op_test_second.csv', index=False)




feature_1 = ['channel', 'trans_amt', 'amt_src1', 'merchant', 'code1', 'code2', 'trans_type1', 'acc_id1',
       'device_code1', 'device_code2', 'device_code3', 'device1', 'device2', 'mac1', 'ip1', 'bal', 
       'amt_src2', 'acc_id2', 'acc_id3','geo_code', 'trans_type2', 'market_code', 'market_type', 'ip1_sub']


for feat in feature_1:
    temp = trans_train.groupby(['UID',feat])['second'].mean().reset_index().rename(columns={'second': feat+'_o_second'})
    trans_train = pd.merge(trans_train, temp, on=['UID',feat], how='left')

trans_train_1 = trans_train.drop(feature_1, axis=1)
trans_train_1 = trans_train_1.drop(['day', 'time', 'second'], axis=1)
trans_train_1.to_csv('trans_train_second.csv', index=False)


for feat in feature_1:
    temp = trans_test.groupby(['UID',feat])['second'].mean().reset_index().rename(columns={'second': feat+'_o_second'})
    trans_test = pd.merge(trans_test, temp, on=['UID',feat], how='left')

trans_test_1 = trans_test.drop(feature_1, axis=1)
trans_test_1 = trans_test_1.drop(['day', 'time', 'second'], axis=1)
trans_test_1.to_csv('trans_test_second.csv', index=False)