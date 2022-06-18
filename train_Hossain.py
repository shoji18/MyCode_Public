import numpy as np
import sys
import csv
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

from Models.model_Hossain import build_model
from tensorflow.keras.callbacks import EarlyStopping

import my_utility as myutil
import config

tf.config.experimental.set_memory_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())

# define constants
LEN_SEQ = config.LEN_SEQ
STRIDE = config.STRIDE
N_CHANNELS = config.N_CHANNELS
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = config.EPOCHS
TRAIN_VALID_RATE = config.TRAIN_VALID_RATE
ES_MONI = config.ES_MONI
ES_MIND = config.ES_MIND
ES_PAT = config.ES_PAT
LOSS = config.LOSS
OPTIMIZER = config.OPTIMIZER

# For intersubject validation
def train_intersub(testdata_name, rootpath):

    # path to save model and prediction results
    modelsave_path = rootpath + "/model_{}.h5".format(testdata_name)
    summary_path = rootpath + "/summary_{}.csv".format(testdata_name)
    tj_period_path = rootpath + "/tj_{}.csv".format(testdata_name)
    tj_peak_path  = rootpath + "/tj_{}.peak.csv".format(testdata_name)

    # build model
    model = build_model(n_channels=16, len_seq=LEN_SEQ)
    
    # prepare dataset
    train_data, train_label, test_data, test_label \
        = myutil.generate_dataset_intersubject(testdata_name)
    train_data = np.asarray(train_data).reshape(-1, N_CHANNELS, LEN_SEQ, 1)
    test_data = np.asarray(test_data).reshape(-1, N_CHANNELS, LEN_SEQ, 1)
    
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

    all_data, all_label= myutil.generate_dataset_crossvalidation()
    all_data = all_data.reshape(-1, N_CHANNELS, LEN_SEQ, 1)

    kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
    es = EarlyStopping(monitor=ES_MONI, min_delta=ES_MIND,
                       verbose=1, patience=ES_PAT)
    
    for cv, (train_idx, test_idx) in enumerate(kf.split(all_data, all_label)):
        
        X_train, X_val, y_train, y_val \
            = train_test_split(all_data[train_idx], all_label[train_idx],
                               test_size=TRAIN_VALID_RATE, shuffle=True)
        model = build_model(n_channels=N_CHANNELS, len_seq=LEN_SEQ)
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