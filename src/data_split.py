# -*- coding: utf-8 -*-
# encoding:utf-8
# @File  : mdata_split.py
# @Author: liushuaipeng
# @Date  : 2018/11/27 15:07
# @Desc  : 数据集划分，95%训练，5%验证
import sys,os

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

current_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(current_path)
print(current_path)

print('0 data spliting')


train_data = pd.read_csv('../data/all/train.csv', sep=',')

title1_zh_list = train_data['title1_zh']#.tolist()
title2_zh_list = train_data['title2_zh']#.tolist()
title1_en_list = train_data['title1_en']#.tolist()
title2_en_list = train_data['title2_en']#.tolist()
lable_list = train_data['label']#.tolist()

print(len(title1_zh_list))
print(len(title2_zh_list))
print(len(title2_en_list))
print(len(title1_en_list))
print(len(lable_list))

title1_zh_len_list = [len(str(i)) for i in title1_zh_list]
title2_zh_len_list = [len(str(i)) for i in title2_zh_list]

title1_zh_max_len = max(title1_zh_len_list)
title2_zh_max_len = max(title2_zh_len_list)
print(title1_zh_max_len,title2_zh_max_len)

title1_zh_total_len = sum(title1_zh_len_list)
title2_zh_total_len = sum(title2_zh_len_list)
print('中文平均長度={}'.format((title1_zh_total_len+title2_zh_total_len)/(len(title1_zh_list)+len(title2_zh_list))))

title1_en_len_list = [len(str(i).split(' ')) for i in title1_en_list]
title2_en_len_list = [len(str(i).split(' ')) for i in title2_en_list]

title1_en_max_len = max(title1_en_len_list)
title2_en_max_len = max(title2_en_len_list)
print('max lenght',title1_en_max_len,title2_en_max_len)
print('min lenght',min(title1_en_len_list),min(title2_en_len_list))
print('mid length',numpy.median(title1_en_len_list),numpy.median(title2_en_len_list))
print('ave length',numpy.mean(title1_en_len_list),numpy.mean(title2_en_len_list))

title1_en_total_len = sum(title1_en_len_list)
title2_en_total_len = sum(title2_en_len_list)
print('english平均長度={}'.format((title1_en_total_len+title2_en_total_len)/(len(title1_en_list)+len(title2_en_list))))



t1_zh_train, t1_zh_dev, t2_zh_train, t2_zh_dev, t1_en_train, t1_en_dev, t2_en_train, t2_en_dev, label_train, label_dev = train_test_split(
    title1_zh_list, title2_zh_list, title1_en_list, title2_en_list, lable_list, test_size=0.05, random_state=666)

# print(len(t1_zh_train))
# print(len(t1_zh_dev))
# print(len(t2_zh_train))
# print(len(t2_zh_dev))
# print(len(t1_en_train))
# print(len(t1_en_dev))
# print(len(t2_en_train))
# print(len(t2_en_dev))
# print(len(label_train))
# print(len(label_dev))

train = pd.DataFrame()
train[0]=label_train
train[1] = t1_zh_train
train[2] = t2_zh_train
train[3] = t1_en_train
train[4] = t2_en_train
train.to_csv('../data/all/train_dataset.csv', sep=',', index=False, header=['label','t1_zh','t2_zh','t1_en','t2_en'])

dev = pd.DataFrame()
dev[0]=label_dev
dev[1] = t1_zh_dev
dev[2] = t2_zh_dev
dev[3] = t1_en_dev
dev[4] = t2_en_dev
dev.to_csv('../data/all/dev_dataset.csv', sep=',', index=False, header=['label','t1_zh','t2_zh','t1_en','t2_en'])
print('0 data split done.')

