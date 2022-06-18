import requests
import numpy as np
import pandas as pd
import csv
import os
import json
import glob
from math import floor
import sklearn.metrics as metrics

import config

DATA_ROOT = config.DATA_ROOT
OLD_DATA = config.OLD_DATA
SAMP_FREQ = config.SAMP_FREQ
OFFSET = config.OFFSET
LEN_SEQ = config.LEN_SEQ
STRIDE = config.STRIDE

def mk_resultdir_intersub(testdata_name, dt_now, modelname="hoge"):
    # Result/
    #     └ eachcase_result
    #            ├ gs_result.csv（each gridsearch result）
    #            ├ model.h5（saved model）
    #            └ summary.csv (auc, f1, and other scores of the best model)

    rootpath = "./Results/{}_{}_{:%H%M%S}".format(modelname, testdata_name, dt_now)
    path_summary = rootpath + "/summary_{}.csv".format(testdata_name)

    os.mkdir(rootpath)
    summary_file = open(path_summary, 'w')
    writer = csv.writer(summary_file, lineterminator='\r')
    data_init = ['Data','AUC','F1','Precision','Recall',
                 'TN','FP','FN','TP']
    writer.writerow(data_init)
    summary_file.close()

    return rootpath


def mk_resultdir_crossval(exp_id, modelname="hoge"):

    rootpath = "./Results/{}_{}".format(modelname, exp_id)
    path_summary = rootpath + "/summary_{}.csv".format(exp_id)

    if os.path.exists(rootpath):
        pass
    else:
        os.mkdir(rootpath)
        summary_file = open(path_summary, 'w')
        writer = csv.writer(summary_file, lineterminator='\r')
        data_init = ['Data','AUC','F1','Precision','Recall',
                    'TN','FP','FN','TP']
        writer.writerow(data_init)
        summary_file.close()

    return rootpath


# generate dataset from data in EDF_NPY and LAB_NPY (for intersubject)
def generate_dataset_intersubject(testdata_name):

    test_data = []
    train_data = []
    train_label = []
    datanames_all = [i.split('/')[1].split('.')[0] 
                     for i in glob.glob('EDF_NPY/*.npy')]

    for dataname in datanames_all:
        label_npy = np.load("LAB_NPY/{}.npy".format(dataname)).astype(int)
        edf_npy = np.load("EDF_NPY/{}.npy".format(dataname))
        
        if dataname == testdata_name:
            print("Testdata:{}".format(dataname))
            for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
                test_data.append(edf_npy[:, j-OFFSET:j+OFFSET])
            test_data = np.asarray(test_data)
            test_label = label_npy[OFFSET:(-1)*OFFSET:STRIDE]
        else:
            for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
                train_data.append(
                    edf_npy[:, j-OFFSET:j+OFFSET])
            train_label.extend(label_npy[OFFSET:(-1)*OFFSET:STRIDE])

    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    print("testdata_shape:", test_data.shape)
    print("testlabel_shape:", test_label.shape)
    print("traindata_shape:", train_data.shape)
    print("trainlabel_shape:", train_label.shape)

    return train_data, train_label, test_data, test_label


# generate dataset from data in EDF_NPY and LAB_NPY (for cross validation)
def generate_dataset_crossvalidation():

    all_data = []
    all_label = []
    datanames_all = [i.split('/')[1].split('.')[0] 
                     for i in glob.glob('EDF_NPY/*.npy')]

    for dataname in datanames_all:
        label_npy = np.load("LAB_NPY/{}.npy".format(dataname)).astype(int)
        edf_npy = np.load("EDF_NPY/{}.npy".format(dataname))

        for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
            all_data.append(edf_npy[:, j-OFFSET:j+OFFSET])
        all_label.extend(label_npy[OFFSET:(-1)*OFFSET:STRIDE])
        
    all_data = np.asarray(all_data)
    all_label = np.asarray(all_label)

    n_pos_all = np.count_nonzero(all_label == 1)
    n_neg_all = np.count_nonzero(all_label == 0)
    print("all_data.shape:", all_data.shape)
    print("all_label.shape:", all_label.shape)
    print("p/n_rate:{}/{}={:.3f}".format(n_pos_all, n_neg_all, 
                                         n_pos_all/n_neg_all))

    return all_data, all_label


def send_notification(message):
    requests.post('Enter URL Here',
                  data = json.dumps({
                      'text': u"{}".format(message), # message to post
                      'username': u'me', # username
                      'icon_emoji': u':ghost:', # emoji
                      'link_names': 1, # validate mention
                  }))


def eval(dataname, true, pred, pred_bi=None):
    
    fpr, tpr, _ = metrics.roc_curve(true.flatten(), pred.flatten())
    auc_val = metrics.auc(fpr, tpr)

    if pred_bi is None:
        pred_bi = np.round(pred)
    else:
        pass

    cm = metrics.confusion_matrix(true.flatten(), pred_bi.flatten())
    f1 = metrics.f1_score(true.flatten(), pred_bi.flatten())

    # culculate pricision
    prec = metrics.precision_score(true.flatten(), pred_bi.flatten())
    # culculate recall
    rec = metrics.recall_score(true.flatten(), pred_bi.flatten())

    print(cm)
    print("F-score", f1)
    print("auc", auc_val)
    
    result = [dataname, auc_val, f1, prec, rec,
              cm[0][0], cm[0][1], cm[1][0], cm[1][1]]

    return result

# generate annotaion file (for TJNoter, area annotation)
def annotation_generator(pred, save_path, sensitivity=0.5):

    HEADER = ["注釈者", "開始時間 (秒)", "終了時間 (秒)", "ラベル",
          "コメント", "チャネル", "参照パターン", "TC", "HF"]
    ANNOTATOR = "Machine"
    LABEL = "Abnormal"
    CHANNELS = "Fp1;Fp2;F3;F4;C3;C4;P3;P4;O1;O2;F7;F8;T3;T4;T5;T6;"
    DAMMY = "dammy"
    OFFSET = config.OFFSET
    LEN_SEQ = config.LEN_SEQ
    SAMP_FREQ = config.SAMP_FREQ

    start_times = []
    end_times = []
    threshold = floor(pred.shape[1]* np.clip((1-sensitivity), 0, 1))

    pred_sum = pred.sum(axis=1)
    pred_bi_1d = np.where(pred_sum > threshold, 1, 0)
    abnormal_indices = pred_bi_1d.nonzero()[0].tolist()

    if len(abnormal_indices) == 0:
        pass
    elif len(abnormal_indices) == 1:
        start_times.append((abnormal_indices[0]*LEN_SEQ+OFFSET)/SAMP_FREQ)
        end_times.append((abnormal_indices[0]*LEN_SEQ+LEN_SEQ+OFFSET)/SAMP_FREQ)
    else:
        start_idx = abnormal_indices[0]
        end_idx = abnormal_indices[0]

        for idx in abnormal_indices[1:]:
            if idx == end_idx+1:
                end_idx = idx
            else:
                start_times.append((start_idx*LEN_SEQ + OFFSET)/SAMP_FREQ)
                end_times.append((end_idx*LEN_SEQ + LEN_SEQ + OFFSET)/SAMP_FREQ)
                start_idx = idx
                end_idx = idx
        
        start_times.append((start_idx*LEN_SEQ + OFFSET) / SAMP_FREQ)
        end_times.append((end_idx*LEN_SEQ + LEN_SEQ + OFFSET) / SAMP_FREQ)

    result = pd.DataFrame(index=range(len(start_times)),
                          columns = HEADER, dtype=str)
    result.astype({'開始時間 (秒)': float, '終了時間 (秒)': float})
    
    for row, (st, ed) in enumerate(zip(start_times, end_times)):
        result.iloc[row].at["注釈者"] = ANNOTATOR
        result.iloc[row].at["開始時間 (秒)"] = st
        result.iloc[row].at["終了時間 (秒)"] = ed
        result.iloc[row].at["ラベル"] = LABEL
        result.iloc[row].at["コメント"] = DAMMY
        result.iloc[row].at["チャネル"] = CHANNELS
        result.iloc[row].at["参照パターン"] = DAMMY
        result.iloc[row].at["TC"] = DAMMY
        result.iloc[row].at["HF"]= DAMMY
    
    result.to_csv(save_path, index=False, encoding="shift-jis")


# generate annotaion file (for TJNoter, point annotation)
def peakfile_generator(pred, save_path):

    datalist = []
    OFFSET = config.OFFSET
    LEN_SEQ = config.LEN_SEQ
    SAMP_FREQ = config.SAMP_FREQ
    
    CH_FILE_PATH = "EDF_NPY/ch_names.txt"
    with open(CH_FILE_PATH) as f:
        ch_names = f.read().splitlines()

    pred_sum = pred.sum(axis=1)
    abnormal_indices = pred_sum.nonzero()[0].tolist()

    if len(abnormal_indices) == 0:
        pass
    else:
        for idx in abnormal_indices:
            for i, [ch_name, data] in enumerate(zip(ch_names, pred[idx])):
                if pred[idx, i] == 1:
                    time_idx = (idx*LEN_SEQ+OFFSET)/SAMP_FREQ
                    datalist.append([ch_name, str(time_idx), "2r"])

    pd.DataFrame(datalist).to_csv(save_path, index=False, encoding="shift-jis")
