import glob
import csv
import os
import cupy
import pickle
import datetime
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split, KFold

import my_utility as myutil
import config
from Models.model_SVM import build_model

LEN_SEQ = config.LEN_SEQ
STRIDE = config.STRIDE
MAX_FREQ = config.MAX_FREQ
OFFSET = config.OFFSET
SEED = config.SEED
N_FOLD = config.N_FOLD
GS_CV = config.GS_CV
GS_METRIC = config.GS_METRIC
SVM_PARAMS = config.SVM_PARAMS

datanames_all = [i.split('/')[1].split('.')[0] 
                 for i in glob.glob('EDF_NPY/*.npy')]


def min_max_scaling_1d(np_data_1d):
    # x' = (x-min(x))/(max(x)-min(x))
    max = np_data_1d.max()
    min = np_data_1d.min()
    scaled_data = (np_data_1d - min) / (max - min)

    return scaled_data


# The input of SVM is 1ch (1-electrode) data.
def data_preprocessing_intersub(testdata_name):

    test_data = []
    test_label = []
    train_data = []
    train_label = []

    for dataname in datanames_all:
        label_npy = np.load("LAB_NPY/{}.npy".format(dataname)).astype(int)
        edf_npy = np.load("EDF_NPY/{}.npy".format(dataname))
        
        if dataname == testdata_name:
            print("Testdata:{}".format(dataname))
            for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
                for k in range(edf_npy.shape[0]):
                    test_data.append(edf_npy[k, j-OFFSET:j+OFFSET])
                    test_label.append(label_npy[j, k])
            test_data = np.asarray(test_data)
            test_data_fft = np.abs(np.fft.fft(test_data))[:,:MAX_FREQ]
            test_label = np.asarray(test_label).flatten()
        else:
            for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
                for k in range(edf_npy.shape[0]):
                    train_data.append(edf_npy[k, j-OFFSET:j+OFFSET])
                    train_label.append(label_npy[j, k])
            
    train_data = np.asarray(train_data)
    train_data_fft = np.abs(np.fft.fft(train_data))[:,:MAX_FREQ]
    train_label = np.asarray(train_label).flatten()

    print(train_data_fft.shape)
    print(test_data_fft.shape)

    return train_data_fft, train_label, test_data_fft, test_label


def data_preprocessing_cv():

    all_data = []
    all_label = []

    for dataname in datanames_all:
        label_npy = np.load("LAB_NPY/{}.npy".format(dataname)).astype(int)
        edf_npy = np.load("EDF_NPY/{}.npy".format(dataname))
        
        for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
            for k in range(edf_npy.shape[0]):
                all_data.append(edf_npy[k, j-OFFSET:j+OFFSET])
                all_label.append(label_npy[j, k])
            
    all_data = np.asarray(all_data)
    all_data_fft = np.abs(np.fft.fft(all_data))[:,:MAX_FREQ]
    all_label = np.asarray(all_label).flatten()

    n_pos_all = np.count_nonzero(all_label == 1)
    n_neg_all = np.count_nonzero(all_label == 0)
    print("all_p/n_rate:{}/{}={:.3f}".format(n_pos_all, n_neg_all, 
                                             n_pos_all/n_neg_all))

    return all_data_fft, all_label


def train_intersub(testdata_name, rootpath):

    summary_path = rootpath + "/summary_{}.csv".format(testdata_name)
    modelsave_path = rootpath + "/model_{}.pkl".format(testdata_name)
    grid_path = rootpath + "/grid_{}.csv".format(testdata_name)

    train_data, train_label, test_data, test_label \
        = data_preprocessing_intersub(testdata_name)
    model = build_model()
    
    clf = GridSearchCV(model, SVM_PARAMS, scoring=GS_METRIC,
                       verbose=2, cv=GS_CV)
    clf.fit(train_data, train_label)
    gs_result = pd.DataFrame.from_dict(clf.cv_results_).to_csv(grid_path)
    best_params = clf.best_params_

    # build new model using best parameters
    # Because in gridsearch, validation data did not be used as train data.
    clf_best = build_model(C=best_params['C'], kernel=best_params['kernel'],
                           gamma=best_params['gamma'], verbose=True)
    clf_best = clf_best.fit(train_data, train_label)

    # save the best model
    pickle.dump(clf_best, open(modelsave_path, 'wb'))

    # Normally, SVM has binary outputs (0 or 1).
    # However, to culculate AUC, outputs need to be continuous value.
    # Therefore I use decision function and assume it as continuous output,
    # (the distance from hyperplane).
    pred = cupy.asnumpy(clf_best.decision_function(test_data).flatten())
    pred_scaled = min_max_scaling_1d(pred)

    result = myutil.eval(testdata_name, test_label, pred_scaled)

    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)
    
    return 0


def train_crossval(rootpath, exp_id):

    summary_path = rootpath + "/summary_{}.csv".format(exp_id)
    grid_path = rootpath + "/grid_{}.csv".format(exp_id)

    with open(summary_path) as f:
        n_exp = (len(f.read().splitlines()) - 1)//5
    print("start {}th experiment".format(n_exp))

    all_data, all_label = data_preprocessing_cv()
    
    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

    for cv, (train_idx, test_idx) in enumerate(kf.split(all_data,
                                                        all_label)):
        
        modelsave_path = rootpath + "/model_{}_{}.h5".format(n_exp, cv)

        n_pos_tr = np.count_nonzero(all_label[train_idx] == 1)
        n_neg_tr = np.count_nonzero(all_label[train_idx] == 0)
        n_pos_te = np.count_nonzero(all_label[test_idx] == 1)
        n_neg_te = np.count_nonzero(all_label[test_idx] == 0)
        print("train_p/n_rate:{}/{}={:.3f}".format(n_pos_tr, n_neg_tr,
                                                   n_pos_tr/n_neg_tr))
        print("test_p/n_rate:{}/{}={:.3f}".format(n_pos_te, n_neg_te,
                                                  n_pos_te/n_neg_te))

        model = build_model()
        
        clf = GridSearchCV(model, SVM_PARAMS, scoring=GS_METRIC,
                           verbose=2, cv=GS_CV)
        clf.fit(all_data[train_idx], all_label[train_idx])
        best_params = clf.best_params_
        print("best_params:", best_params)

        clf_best = build_model(C=best_params['C'], kernel=best_params['kernel'],
                               gamma=best_params['gamma'], verbose=True)
        clf_best.fit(all_data[train_idx], all_label[train_idx])

        pickle.dump(clf_best, open(modelsave_path, 'wb'))

        pred = clf_best.decision_function(all_data[test_idx]).flatten()
        pred = cupy.asnumpy(pred)
        pred_scaled = min_max_scaling_1d(pred)
        pred_bi = clf.predict(all_data[test_idx]).flatten()

        result = myutil.eval("cv{}_{}".format(n_exp, cv),
                             all_label[test_idx], pred_scaled, pred_bi=pred_bi)

        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)
        with open(grid_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["cv{}_{}".format(n_exp, cv), best_params['C'],
                             best_params['gamma']])
    

if __name__ == '__main__':
    codename = sys.argv[0].split('.')[0]
    dt_now = datetime.datetime.now()

    # For intersubject validation
    # python train.py <testdata_name> [Terminal]
    #"""
    testdata_name = sys.argv[1]
    rootpath = myutil.mk_resultdir_intersub(testdata_name, dt_now, codename)
    train_intersub(testdata_name, rootpath)
    #"""
    
    # For N-fold cross validation
    # python train.py <exp_id> [Terminal]
    """
    exp_id = sys.argv[1]
    rootpath = myutil.mk_resultdir_crossval(exp_id, codename)
    train_crossval(rootpath, exp_id=exp_id)
    """