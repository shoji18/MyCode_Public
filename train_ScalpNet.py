import numpy as np
import sys
import csv
import glob
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

from Models.model_ScalpNet import build_model
from tensorflow.keras.callbacks import EarlyStopping

import my_utility as myutil
# from losses import focal_loss, cb_focal_loss
import config

tf.config.experimental.set_memory_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())

# define constants
LEN_SEQ = config.LEN_SEQ
STRIDE = config.STRIDE
OFFSET = config.OFFSET
N_CHANNELS = config.N_CHANNELS
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
TRAIN_VALID_RATE = config.TRAIN_VALID_RATE
ES_MONI = config.ES_MONI
ES_MIND = config.ES_MIND
ES_PAT = config.ES_PAT
LOSS = config.LOSS
OPTIMIZER = config.OPTIMIZER


def setEEG(data):
    map = {0: [0, 1], 1: [1, 1], 2: [2, 1], 3: [3, 1], 4: [4, 1], 5: [1, 0],
           6: [2, 0], 7: [3, 0], 8: [0, 2], 9: [1, 2],10: [2, 2], 11: [3, 2],
           12: [4, 2], 13: [1, 3], 14: [2, 3], 15: [3, 3]}
    x = np.zeros([5, 4, data.shape[1]])
    for i in range(16):
        x[map[i][0], map[i][1], :] = data[i, :]

    return x

# ScalpNet has special preprocessing method
def data_preprocessing_intersub(testdata_name):

    test_data = []
    train_data = []
    train_label = []
    datanames_all = [i.split('/')[1].split('.')[0] 
                     for i in glob.glob('EDF_NPY/*.npy')]

    for dataname in datanames_all:
        label_npy = np.load("LAB_NPY/{}.npy".format(dataname)).astype(int)
        edf_npy = setEEG(np.load("EDF_NPY/{}.npy".format(dataname)))

        if dataname == testdata_name:
            print("Testdata:{}".format(dataname))
            for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
                test_data.append(edf_npy[:, :, j-OFFSET:j+OFFSET] * 10)
            print(len(test_data))
            test_data = np.asarray(test_data)
            test_label = label_npy[OFFSET:(-1)*OFFSET:STRIDE]
        else:
            for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
                train_data.append(edf_npy[:, :, j-OFFSET:j+OFFSET] * 10)
            train_label.extend(label_npy[OFFSET:(-1)*OFFSET:STRIDE])

    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)

    print("datashape:", test_data.shape)
    print("labelshape:", test_label.shape) 
    print("datashape:", train_data.shape)
    print("labelshape:", train_label.shape)

    return train_data, train_label, test_data, test_label


def data_preprocessing_cv():

    all_data = []
    all_label = []
    datanames_all = [i.split('/')[1].split('.')[0] 
                     for i in glob.glob('EDF_NPY/*.npy')]

    for dataname in datanames_all:
        label_npy = np.load("LAB_NPY/{}.npy".format(dataname)).astype(int)
        edf_npy = np.load("EDF_NPY/{}.npy".format(dataname))
        edf_npy = setEEG(edf_npy)
        print(edf_npy.shape)
        print(label_npy.shape)

        for j in range(OFFSET, label_npy.shape[0]-OFFSET, STRIDE):
            all_data.append(edf_npy[:, :, j-OFFSET:j+OFFSET] * 10)
        all_label.extend(label_npy[OFFSET:(-1)*OFFSET:STRIDE])
        
    all_data = np.asarray(all_data)
    all_label = np.asarray(all_label)
    print("datashape:", all_data.shape)
    print("labelshape:", all_label.shape)

    n_pos_all = np.count_nonzero(all_label == 1)
    n_neg_all = np.count_nonzero(all_label == 0)
    print("all_p/n_rate:{}/{}".format(n_pos_all, n_neg_all))

    return all_data, all_label


# For intersubject validation
def train_intersub(testdata_name, rootpath):

    # path to save model and prediction results
    modelsave_path = rootpath + "/model_{}.h5".format(testdata_name)
    summary_path = rootpath + "/summary_{}.csv".format(testdata_name)
    tj_period_path = rootpath + "/tj_{}.csv".format(testdata_name)
    tj_peak_path  = rootpath + "/tj_{}.peak.csv".format(testdata_name)

    # build model
    model = build_model()
    
    # prepare dataset
    train_data, train_label, test_data, test_label \
        = data_preprocessing_intersub(testdata_name)
    
    # prepare validation data
    X_train, X_val, y_train, y_val \
        = train_test_split(train_data, train_label,
                           test_size=TRAIN_VALID_RATE, shuffle=True)

    # set early stopping following config file
    es = EarlyStopping(monitor=ES_MONI, min_delta=ES_MIND,
                       verbose=1, patience=ES_PAT)
    model.compile(optimizer=OPTIMIZER, loss=LOSS)

    print("X_train.shape:{}, y_train.shape:{}".format(X_train.shape,
                                                      y_train.shape))

    model.fit(X_train, y_train, epochs=EPOCHS,
              batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
              callbacks=[es])
    model.save(modelsave_path)

    # prediction
    pred = model.predict(test_data)

    # generate annotaion file (for TJNoter, area annotation)
    myutil.annotation_generator(pred.round(), tj_period_path)

    # generate annotaion file (for TJNoter, point annotation)
    myutil.peakfile_generator(pred.round(), tj_peak_path)
    
    # evaluate results 
    result = myutil.eval(testdata_name, test_label, pred)

    with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)


# For N-fold cross validation
# ----- Notice -----
# To do N-fold crossvalidation multiple times with different random seed,
# change config.SEED and keep the "same" exp_id.
# exp) SEED=1000 [config.py] => train_crossval(<path>, 0)
#      SEED=1001 [config.py] => train_crossval(<path>, 0)
# To do this, all results will be written in the same summary file.

def train_crossval(rootpath, exp_id=0):

    N_FOLD = config.N_FOLD
    SEED = config.SEED

    summary_path = rootpath + "/summary_{}.csv".format(exp_id)

    # for multiple experiments
    with open(summary_path) as f:
        n_exp = (len(f.read().splitlines()) - 1) // N_FOLD
    print("start {}th experiment".format(n_exp))

    all_data, all_label= data_preprocessing_cv()

    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    es = EarlyStopping(monitor=ES_MONI, min_delta=ES_MIND,
                       verbose=1, patience=ES_PAT)
    
    for cv, (train_idx, test_idx) in enumerate(kf.split(all_data, all_label)):
        
        X_train, X_val, y_train, y_val \
            = train_test_split(all_data[train_idx], all_label[train_idx],
                               test_size=TRAIN_VALID_RATE, shuffle=True)
        model = build_model()
        model.compile(optimizer=OPTIMIZER, loss=LOSS)

        model.fit(X_train, y_train, epochs=EPOCHS,
                batch_size=BATCH_SIZE, validation_data=(X_val, y_val),
                callbacks=[es])

        model.save(rootpath + "/model_{}_{}.h5".format(n_exp, cv))

        pred = model.predict(all_data[test_idx])
        result = myutil.eval("cv{}_{}".format(n_exp, cv),
                             all_label[test_idx], pred)

        with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)


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