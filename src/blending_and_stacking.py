# encoding:utf-8
from __future__ import division
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import codecs
import numpy as np
import sys
import os
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
reload(sys)
sys.setdefaultencoding('utf8')
import warnings
warnings.filterwarnings("ignore")

# try:
#     import cPickle as pickle
# except:
#     import pickle

ma = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
am = {0: 'agreed', 1: 'disagreed', 2: 'unrelated'}


def weighted_acc_score(y_true, y_test):
    w = {}
    w[0] = 1.0 / 15
    w[1] = 1.0 / 5
    w[2] = 1.0 / 16
    sample_weight = [w[i] for i in y_true]
    weighted_acc = accuracy_score(y_true, y_test, normalize=True, sample_weight=sample_weight)
    return weighted_acc


w_acc_score = metrics.make_scorer(weighted_acc_score, greater_is_better=True)
scoring = {'w_acc': w_acc_score}


class Stacking:
    def __init__(self, eval_prob_path, pred_prob_path, dev_filepath, test_filepath, save_filepath):
        self.eval_prob_path = eval_prob_path
        self.pred_prob_path = pred_prob_path
        self.dev_filepath = dev_filepath
        self.test_filepath = test_filepath
        self.save_filepath = save_filepath

    def load_data(self, filepath):
        data = pd.read_csv(filepath, encoding='utf-8').fillna(' ', inplace=False)
        return data

    def get_feature_from_dir(self, data_path, topk):
        feature_list = []
        name_list = os.listdir(data_path)
        name_list.sort(reverse=True)
        name_list = name_list[:topk]
        for name in name_list:
            filepath = os.path.join(data_path, name)
            temp_list = []
            with codecs.open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line_l = line.strip().split('\t')
                    line_l = [float(x) for x in line_l]
                    temp_list.append(line_l)
            temp_feature = np.vstack(temp_list)
            feature_list.append(temp_feature)
        feature = np.hstack(feature_list)
        return feature, name_list

    def get_stacking_data(self, topk=10):
        print('get val feature')
        feature, use_val_name_list = self.get_feature_from_dir(self.eval_prob_path, topk)
        print('get test feature')
        feature_test, use_test_name_list = self.get_feature_from_dir(self.pred_prob_path, topk)

        for x, y in zip(use_val_name_list, use_test_name_list):
            print(x, y)

        # label:
        labels = self.load_data(self.dev_filepath).label.values
        labels = np.array([ma[i] for i in labels])
        print('len labels:{}'.format(len(labels)))
        return feature, labels, feature_test

    def grid_search_knn(self, X, Y, param_dict):
        print('grid_search for KNeighborsClassifier...')
        params = {
            'n_neighbors': [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 500]
        }
        estimator = KNeighborsClassifier()
        gridsearch = GridSearchCV(estimator, param_grid=params, n_jobs=-1, scoring=scoring, cv=5, verbose=0,
                                  refit='w_acc')
        gridsearch.fit(X, Y)
        print(gridsearch.best_params_, gridsearch.best_score_)
        param_dict['knn_n_neighbors'] = gridsearch.best_params_['n_neighbors']

    def grid_search_softmax(self, X, Y, param_dict):
        print('grid_search for softmax...')
        params = {
            'C': [0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.038, 0.04, 0.045, 0.05, 0.1],
        }
        estimator = LogisticRegression(multi_class='ovr', class_weight='balanced',solver='lbfgs')
        gridsearch = GridSearchCV(estimator, param_grid=params, n_jobs=-1, scoring=scoring, cv=5, verbose=0,
                                  refit='w_acc')
        gridsearch.fit(X, Y)
        print(gridsearch.best_params_, gridsearch.best_score_)
        param_dict['softmax_C'] = gridsearch.best_params_['C']
        # param_dict['softmax_solver'] = gridsearch.best_params_['solver']

    def grid_search_svm(self, X, Y, param_dict):
        print('grid_search for svm...')
        print('linear svm...')
        params = {
            'C': [0.007, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.3],
        }
        estimator = svm.SVC(kernel='linear', degree=3, gamma='auto',
                            coef0=0.0, shrinking=True, probability=False,
                            tol=1e-3, cache_size=200, class_weight='balanced',
                            verbose=False, max_iter=-1, random_state=None, decision_function_shape='ovr')

        gridsearch = GridSearchCV(estimator, param_grid=params, n_jobs=-1, scoring=scoring, cv=5, verbose=0,
                                  refit='w_acc')
        gridsearch.fit(X, Y)
        print(gridsearch.best_params_, gridsearch.best_score_)
        param_dict['linear_svm_C'] = gridsearch.best_params_['C']

        print('rbf svm...')
        params = {
            'C': [0.007, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.1, 0.3],
        }
        estimator = svm.SVC(kernel='rbf', degree=3, gamma='auto',
                            coef0=0.0, shrinking=True, probability=False,
                            tol=1e-3, cache_size=200, class_weight='balanced',
                            verbose=False, max_iter=-1, random_state=None, decision_function_shape='ovr')

        gridsearch = GridSearchCV(estimator, param_grid=params, n_jobs=-1, scoring=scoring, cv=5, verbose=0,
                                  refit='w_acc')
        gridsearch.fit(X, Y)
        print(gridsearch.best_params_, gridsearch.best_score_)
        param_dict['rbf_svm_C'] = gridsearch.best_params_['C']

        print('poly svm...')
        params = {
            'C': [0.007, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.1, 0.3],
            'degree': [2, 3, 4]
        }
        estimator = svm.SVC(kernel='poly', gamma='auto',
                            coef0=0.0, shrinking=True, probability=False,
                            tol=1e-3, cache_size=200, class_weight='balanced',
                            verbose=False, max_iter=-1, random_state=None, decision_function_shape='ovr')
        gridsearch = GridSearchCV(estimator, param_grid=params, n_jobs=-1, scoring=scoring, cv=5, verbose=0,
                                  refit='w_acc')
        gridsearch.fit(X, Y)
        print(gridsearch.best_params_, gridsearch.best_score_)
        param_dict['poly_svm_C'] = gridsearch.best_params_['C']
        param_dict['poly_svm_degree'] = gridsearch.best_params_['degree']

    def pipline(self, topk=20, random_state=88, search=False):
        '''

        :param topk:
        :return:
        '''
        # get data=============================================================================

        X, Y, test_data = self.get_stacking_data(topk=topk)

        # get param
        param_dict = {}
        if search:
            self.grid_search_softmax(X, Y, param_dict)
            self.grid_search_knn(X, Y, param_dict)
            self.grid_search_svm(X, Y, param_dict)
        else:
            param_dict['knn_n_neighbors'] = 500
            param_dict['softmax_C'] = 0.008
            param_dict['linear_svm_C'] = 0.007
            param_dict['rbf_svm_C'] = 0.02
            param_dict['poly_svm_C'] = 0.01
            param_dict['poly_svm_degree'] = 2
            # param_dict['knn_n_neighbors'] = 300
            # param_dict['softmax_C'] = 0.007
            # param_dict['linear_svm_C'] = 0.007
            # param_dict['rbf_svm_C'] = 0.008
            # param_dict['poly_svm_C'] = 0.015
            # param_dict['poly_svm_degree'] = 2
        print(param_dict)
        # stacking=============================================================================
        nfold = 5
        skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=random_state)
        iterator = 1
        second_labels = []
        second_labels_predict = []
        second_features = []
        test_features = []
        for train_index, test_index in skf.split(X, Y):
            temp_fea_list = []
            temp_fea_list_test = []

            print('**************', iterator, '************************')
            iterator += 1
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]

            second_labels.extend(y_test)  # nfold 份

            # softmax
            print('softmax...')
            softmax_model = LogisticRegression(multi_class="ovr", solver='lbfgs', C=param_dict['softmax_C'],
                                               class_weight='balanced')
            softmax_model.fit(X_train, y_train)
            softmax_fea = softmax_model.predict_proba(X_test)
            temp_fea_list.append(softmax_fea)
            temp_fea_list_test.append(softmax_model.predict_proba(test_data))

            # MultinomialNB
            print('mul_nb...')
            mul_nb = MultinomialNB()
            mul_nb.fit(X_train, y_train)
            mul_nb_fea = mul_nb.predict_proba(X_test)
            temp_fea_list.append(mul_nb_fea)
            temp_fea_list_test.append(mul_nb.predict_proba(test_data))

            # knn
            print('knn..')
            knn_model = KNeighborsClassifier(n_neighbors=param_dict['knn_n_neighbors'])
            knn_model.fit(X_train, y_train)
            knn_fea = knn_model.predict_proba(X_test)
            temp_fea_list.append(knn_fea)
            temp_fea_list_test.append(knn_model.predict_proba(test_data))

            # svm linear
            print('svm linear...')
            svm_model = svm.SVC(C=param_dict['linear_svm_C'], kernel='linear', degree=3, gamma='auto',
                                coef0=0.0, shrinking=True, probability=True,
                                tol=1e-3, cache_size=200, class_weight='balanced',
                                verbose=False, max_iter=-1, decision_function_shape='ovr',
                                random_state=None)
            svm_model.fit(X_train, y_train)
            svm_linear_fea = svm_model.predict_proba(X_test)
            temp_fea_list.append(svm_linear_fea)
            temp_fea_list_test.append(svm_model.predict_proba(test_data))

            # svm rbf
            print('svm rbf...')
            svm_model = svm.SVC(C=param_dict['rbf_svm_C'], kernel='rbf', degree=3, gamma='auto',
                                coef0=0.0, shrinking=True, probability=True,
                                tol=1e-3, cache_size=200, class_weight='balanced',
                                verbose=False, max_iter=-1, decision_function_shape='ovr',
                                random_state=None)
            svm_model.fit(X_train, y_train)
            svm_rbf_fea = svm_model.predict_proba(X_test)
            temp_fea_list.append(svm_rbf_fea)
            temp_fea_list_test.append(svm_model.predict_proba(test_data))

            # svm poly
            print('svm poly...')
            svm_model = svm.SVC(C=param_dict['poly_svm_C'], kernel='poly', degree=param_dict['poly_svm_degree'],
                                gamma='auto',
                                coef0=0.0, shrinking=True, probability=True,
                                tol=1e-3, cache_size=200, class_weight='balanced',
                                verbose=False, max_iter=-1, decision_function_shape='ovr',
                                random_state=None)
            svm_model.fit(X_train, y_train)
            svm_poly_fea = svm_model.predict_proba(X_test)
            temp_fea_list.append(svm_poly_fea)
            temp_fea_list_test.append(svm_model.predict_proba(test_data))

            print('len temp_fea_list:{}'.format(len(temp_fea_list)))
            print('len temp_fea_list_test:{}'.format(len(temp_fea_list_test)))
            temp_sum_fea = np.zeros_like(temp_fea_list[0])
            for d in temp_fea_list:
                temp_sum_fea += d
            temp_sum_fea /= len(temp_fea_list)
            print('temp_sum_fea:{}'.format(temp_sum_fea.shape))
            temp_labels_predict = np.argmax(temp_sum_fea, axis=1)
            second_labels_predict.extend(temp_labels_predict)

            # merge
            temp_fea = np.hstack(temp_fea_list)
            temp_fea_test = np.hstack(temp_fea_list_test)
            test_features.append(temp_fea_test)
            second_features.append(temp_fea)

        res_test_feature = np.zeros_like(test_features[0])
        for d in test_features:
            res_test_feature += d
        res_test_feature /= len(test_features)
        res_fea = np.vstack(second_features)
        res_labels = np.array(second_labels)
        print(res_test_feature.shape)
        print(res_fea.shape)
        print(len(res_labels))
        # last softmax===================================================================================
        X = res_fea
        Y = res_labels
        X_for_test = res_test_feature

        test_data = self.load_data(self.test_filepath)
        test_ids = test_data.id.values

        if search:
            self.grid_search_softmax(X, Y, param_dict)
        else:
            param_dict['softmax_C'] = 0.02
        estimator = LogisticRegression(C=param_dict['softmax_C'], multi_class='ovr', solver='lbfgs',
                                       class_weight='balanced')
        estimator.fit(X, Y)
        predict_labels = estimator.predict(X_for_test)
        predict_labels = [am[x] for x in predict_labels]

        res_data = pd.DataFrame({'Id': test_ids, 'Category': predict_labels})
        res_data.to_csv(self.save_filepath, index=False)


if __name__ == '__main__':
    print('start stacking...')
    if len(sys.argv) != 5:
        print('请确认参数数量是否正确')
    else:
        eval_prob_path = sys.argv[1]
        pred_prob_path = sys.argv[2]
        topK = int(sys.argv[3])
        search = True if (sys.argv[4]).strip().lower() == 'true' else False
        print('eval_prob_path='+ eval_prob_path)
        print('pred_prob_path='+pred_prob_path)
        print('topK='+str(topK))
        print('search='+str(search))
        st = Stacking(eval_prob_path=eval_prob_path,
                      pred_prob_path=pred_prob_path,
                      dev_filepath='../data/all/dev_dataset.csv',
                      test_filepath='../data/all/test.csv',
                      save_filepath='../result/result.csv')
        st.pipline(topk=topK, random_state=888, search=search)
    '''
    st = Stacking(eval_prob_path='../data/eval_prob_exist',
                  pred_prob_path='../data/pred_prob_exist',
                  dev_filepath='../data/all/dev_dataset.csv',
                  test_filepath='../data/all/test.csv',
                  save_filepath='../result/result.csv')
    st.pipline(topk=25, random_state=888, search=False)
    '''