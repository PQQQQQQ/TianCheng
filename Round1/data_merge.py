import pandas as pd
import numpy as np
import warnings
import gc

warnings.filterwarnings("ignore")


tag_train = pd.read_csv('../tag_train_new.csv')
sub = pd.read_csv('../sub.csv')


cnt_op_train = pd.read_csv('cnt_op_train.csv')
cnt_op_test = pd.read_csv('cnt_op_test.csv')
cnt_trans_train = pd.read_csv('cnt_trans_train.csv')
cnt_trans_test = pd.read_csv('cnt_trans_test.csv')

op_train_inv = pd.read_csv('op_train_inv.csv')
op_test_inv = pd.read_csv('op_test_inv.csv')
trans_train_inv = pd.read_csv('trans_train_inv.csv')
trans_test_inv = pd.read_csv('trans_test_inv.csv')

count_op_train = pd.read_csv('count_op_train.csv')
count_op_test = pd.read_csv('count_op_test.csv')
count_trans_train = pd.read_csv('count_trans_train.csv')
count_trans_test = pd.read_csv('count_trans_test.csv')

n_count_op_train = pd.read_csv('n_count_op_train.csv')
n_count_op_test = pd.read_csv('n_count_op_test.csv')
n_count_trans_train = pd.read_csv('n_count_trans_train.csv')
n_count_trans_test = pd.read_csv('n_count_trans_test.csv')

# op_train_nunique = pd.read_csv('op_train_nunique_1.csv')
# op_test_nunique = pd.read_csv('op_test_nunique_1.csv')
#trans_train_nunique = pd.read_csv('trans_train_nunique.csv')
#trans_test_nunique = pd.read_csv('trans_test_nuniuqe.csv')

gc.collect()

op_train_mean = n_count_op_train.groupby(['UID']).mean().reset_index()
#op_train_sum = n_count_op_train.groupby(['UID']).sum().reset_index()
op_train_max = n_count_op_train.groupby(['UID']).max().reset_index()
#op_train_min = n_count_op_train.groupby(['UID']).min().reset_index()
#op_train_std = n_count_op_train.groupby(['UID']).std().reset_index()
#op_train_median = n_count_op_train.groupby(['UID']).median().reset_index()
#op_train_skew = n_count_op_train.groupby(['UID']).skew().reset_index()


trans_train_mean = n_count_trans_train.groupby(['UID']).mean().reset_index()
#trans_train_sum = n_count_trans_train.groupby(['UID']).sum().reset_index()
trans_train_max = n_count_trans_train.groupby(['UID']).max().reset_index()
#trans_train_min = n_count_trans_train.groupby(['UID']).min().reset_index()
#trans_train_std = n_count_trans_train.groupby(['UID']).std().reset_index()
trans_train_median = n_count_trans_train.groupby(['UID']).median().reset_index()
#trans_train_skew = n_count_trans_train.groupby(['UID']).skew().reset_index()


#op_train_nunique_mean = op_train_nunique.groupby(['UID']).mean().reset_index()
#op_train_nunique_sum = op_train_nunique.groupby(['UID']).sum().reset_index()
#op_train_nunique_max = op_train_nunique.groupby(['UID']).max().reset_index()
#op_train_nunique_min = op_train_nunique.groupby(['UID']).min().reset_index()
#op_train_nunique_median = op_train_nunique.groupby(['UID']).median().reset_index()
#op_train_nunique_skew = op_train_nunique.groupby(['UID']).skew().reset_index()

gc.collect()


op_test_mean = n_count_op_test.groupby(['UID']).mean().reset_index()
#op_test_sum = n_count_op_test.groupby(['UID']).sum().reset_index()
op_test_max = n_count_op_test.groupby(['UID']).max().reset_index()
#op_test_min = n_count_op_test.groupby(['UID']).min().reset_index()
#op_test_std = n_count_op_test.groupby(['UID']).std().reset_index()
#op_test_median = n_count_op_test.groupby(['UID']).median().reset_index()
#op_test_skew = n_count_op_test.groupby(['UID']).skew().reset_index()


trans_test_mean = n_count_trans_test.groupby(['UID']).mean().reset_index()
#trans_test_sum = n_count_trans_test.groupby(['UID']).sum().reset_index()
trans_test_max = n_count_trans_test.groupby(['UID']).max().reset_index()
#trans_test_min = n_count_trans_test.groupby(['UID']).min().reset_index()
#trans_test_std = n_count_trans_test.groupby(['UID']).std().reset_index()
trans_test_median = n_count_trans_test.groupby(['UID']).median().reset_index()
#trans_test_skew = n_count_trans_test.groupby(['UID']).skew().reset_index()


#op_test_nunique_mean = op_test_nunique.groupby(['UID']).mean().reset_index()
#op_test_nunique_sum = op_test_nunique.groupby(['UID']).sum().reset_index()
#op_test_nunique_max = op_test_nunique.groupby(['UID']).max().reset_index()
#op_test_nunique_min = op_test_nunique.groupby(['UID']).min().reset_index()
#op_test_nunique_median = op_test_nunique.groupby(['UID']).median().reset_index()
#op_test_nunique_skew = op_test_nunique.groupby(['UID']).skew().reset_index()

gc.collect()


train = pd.merge(tag_train, cnt_op_train, on='UID', how='left')
train = pd.merge(train, cnt_trans_train, on='UID', how='left')
train = pd.merge(train, op_train_inv, on='UID', how='left')
train = pd.merge(train, count_op_train, on='UID', how='left')
train = pd.merge(train, count_trans_train, on='UID', how='left')
train = pd.merge(train, trans_train_inv, on='UID', how='left')
train = pd.merge(train, op_train_mean, on='UID', how='left')
train = pd.merge(train, op_train_max, on='UID', how='left')
train = pd.merge(train, trans_train_mean, on='UID', how='left')
train = pd.merge(train, trans_train_max, on='UID', how='left')
train = pd.merge(train, trans_train_median, on='UID', how='left')
# train = pd.merge(train, op_train_nunique, on='UID', how='left')



test = pd.merge(sub, cnt_op_test, on='UID', how='left')
test = pd.merge(test, cnt_trans_test, on='UID', how='left')
test = pd.merge(test, op_test_inv, on='UID', how='left')
test = pd.merge(test, count_op_test, on='UID', how='left')
test = pd.merge(test, count_trans_test, on='UID', how='left')
test = pd.merge(test, trans_test_inv, on='UID', how='left')
test = pd.merge(test, op_test_mean, on='UID', how='left')
test = pd.merge(test, op_test_max, on='UID', how='left')
test = pd.merge(test, trans_test_mean, on='UID', how='left')
test = pd.merge(test, trans_test_max, on='UID', how='left')
test = pd.merge(test, trans_test_median, on='UID', how='left')
# test = pd.merge(test, op_test_nunique, on='UID', how='left')



train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)