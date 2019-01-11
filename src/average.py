# -*- coding: utf-8 -*-
# encoding:utf-8
# @File  : maverage.py
# @Author: liushuaipeng
# @Date  : 2019/1/7 15:11
# @Desc  : 日志挖掘程序入口
import sys

import numpy
from sklearn import metrics

import pandas,os
import numpy as np

def ave_eval_prob():
    basedir = '../data/eval_prob_exist'
    result = []
    files = [
    ]
    files = os.listdir(basedir)

    for file in files:
        path = os.path.join(basedir, file)
        weight = 1.0
        # weight = float(file.split('tsv.')[-1])
        print('weighted acc:', weight)
        matrix = numpy.loadtxt(path, delimiter='\t')
        result.append(matrix * weight)
    all = numpy.zeros((result[0].shape[0], result[0].shape[1]))
    for m in result:
        all += m

    result = numpy.argmax(all, axis=1)
    tag = ["agreed", "disagreed", "unrelated"]
    pred_y = [tag[i] for i in result]
    # print('pred_y:',pred_y[:10])

    eval_data = pandas.read_csv('../data/all/dev_dataset.csv', sep=',')
    true_y = eval_data['label'].tolist()
    # print('true_y:',true_y[:10])
    w = {}
    w["agreed"] = 1.0 / 15
    w["disagreed"] = 1.0 / 5
    w['unrelated'] = 1.0 / 16
    sample_weight = [w[i] for i in true_y]
    weighted_acc = metrics.accuracy_score(true_y, pred_y, normalize=True, sample_weight=sample_weight)
    print(metrics.confusion_matrix(true_y, pred_y))
    print(metrics.classification_report(true_y, pred_y))
    print('********************')
    print('ensemble weighted_acc=', weighted_acc)

def ave_pred_prob():
    basedir = '../data/pred_prob_exist'
    result = []
    files = []
    files = os.listdir(basedir)
    for file in files:
        weight= 1.0
        weight = float(file.split('tsv.')[-1])
        matrix = np.loadtxt(os.path.join(basedir, file), delimiter='\t')
        result.append(matrix * weight)
    all = np.zeros((result[1].shape[0], result[0].shape[1]))
    for m in result:
        all += m
    result = np.argmax(all, axis=1)
    print(result[:20])
    tag = ["agreed", "disagreed", "unrelated"]
    lables = [tag[i] for i in result]
    train_data = pandas.read_csv('../data/all/test.csv', sep=',')
    ID = train_data['id'].tolist()
    print(len(ID))

    output = pandas.DataFrame()
    output[0] = ID
    output[1] = lables
    output.to_csv('../result/result_avg.csv', sep=',', index=False, header=['Id', 'Category'])
    print('./result_avg.csv')
if __name__=='__main__':
    ave_eval_prob()
    ave_pred_prob()