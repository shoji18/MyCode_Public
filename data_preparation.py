import numpy as np
import os
from mne.io import read_raw_edf
import pandas as pd
import glob
from tqdm import tqdm
from tabulate import tabulate
import config

# this code generates .npy array from .edf(EEG data) and .csv(Label data)
# output shape EDF:(ch, time_length)  Label:(time_length, ch)

DATA_ROOT = config.DATA_ROOT
OLD_DATA = config.OLD_DATA
SAMP_FREQ = config.SAMP_FREQ
RECTWAVE_TIME = config.RECTWAVE_TIME
CH_LEFT = config.CH_LEFT
CH_RIGHT = config.CH_RIGHT

def dir_check():
    if not os.path.isdir("EDF_NPY"):
        os.makedirs("EDF_NPY")
    if not os.path.isdir("LAB_NPY"):
        os.makedirs("LAB_NPY")
    if not os.path.isdir("Results"):
        os.makedirs("Results")


#EDF to numpy
def edf_to_npy():
    edf_paths = glob.glob(DATA_ROOT + 'JAE01_EDF/*')
    label_paths = glob.glob(DATA_ROOT + 'JAE01_label/*')
    names = [i.split('/')[7].split('.')[0] for i in label_paths]
    print("found {} edf data".format(len(edf_paths)))
    print("found {} label data".format(len(names)))
    CH_LEFT = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']
    CH_RIGHT = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']

    for k in tqdm(edf_paths):
        if k.split('/')[7].split('.')[0] in names:
            eeg = []
            ch_names = []

            raw = read_raw_edf(k, verbose='WARNING', preload = True)

            # check samp_freq of loaded EDF
            if raw.info["sfreq"] != SAMP_FREQ:
                raise Exception("sampling frequency of {} is not 500Hz!"
                                .format(k.split('/')[7].split('.')[0]))

            data = raw.get_data()[:, RECTWAVE_TIME:-RECTWAVE_TIME]

            bias_l = data[raw.info["ch_names"].index("A1")]
            bias_r = data[raw.info["ch_names"].index("A2")]

            for n, i in enumerate(raw.info["ch_names"]):
                if i in CH_LEFT:
                    eeg.append(data[n] - bias_l)
                    ch_names.append(i)
                elif i in CH_RIGHT:
                    eeg.append(data[n] - bias_r)
                    ch_names.append(i)

            eeg = np.asarray(eeg)
            np.save('EDF_NPY/' + k.split('/')[7].split('.')[0], eeg)

            with open("EDF_NPY/ch_names.txt", mode='wt') as f:
                for ele in ch_names:
                    f.write(ele + '\n')


def label_to_npy():
    channels = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5',
                'Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T4', 'T6']
    paths = glob.glob(DATA_ROOT + 'JAE01_label/*')

    # convert labelfile (csv, second) to array (ndarray, index)
    print("Loading labels ...")
    for i in tqdm(paths):
        dataname = i.split("/")[7].split(".")[0]
        label_csv = pd.read_csv(i, encoding="SHIFT-JIS").values
        edf_npy = np.load("EDF_NPY/" + dataname + ".npy")
        label_npy = np.zeros([int(edf_npy.shape[1]), 16])
        label_type = label_csv[:, 3].astype(str)
        if dataname in OLD_DATA:
            abnormal_indices = ((label_csv[:, 1:3]) * SAMP_FREQ).astype(int)
            # artifacts is not abnormal 
            abnormal_indices = np.delete(abnormal_indices,
                                            np.where(label_type=="Artifact"), axis=0)
            # make label ndarray
            for j in range(abnormal_indices.shape[0]):
                abn_chs = str(label_csv[j, 5]).split(";")[:-1]
                tmp = np.asarray([1 if i in abn_chs else 0 for i in channels])
                for k in range(abnormal_indices[j,0]-RECTWAVE_TIME, 
                            abnormal_indices[j,1]-RECTWAVE_TIME):
                    label_npy[k,:] = tmp
        elif len(label_csv) < 2: 
            pass
        else:
            abnormal_indices = ((label_csv[1:,1:3].astype(float))
                                 * SAMP_FREQ).astype(int)

            # artifacts is not abnormal 
            abnormal_indices = np.delete(abnormal_indices,
                                         np.where(label_type=="Artifact"),
                                         axis=0)
            
            # make label ndarray
            for j in range(abnormal_indices.shape[0]):
                abn_chs = str(label_csv[j+1, 5]).split(";")[:-1]
                tmp = np.asarray([1 if i in abn_chs else 0 for i in channels])
                for k in range(abnormal_indices[j,0]-RECTWAVE_TIME, 
                               abnormal_indices[j,1]-RECTWAVE_TIME):
                    label_npy[k,:] = tmp
            
        np.save("LAB_NPY/"+i.split("/")[7].split(".")[0]+".npy", label_npy)


# show information of loaded edf/label data on command line
def show_data_info():
    table = []
    headers = ["dataname", "edf_shape", "label_shape", "normal", "abnormal"]
    names = [i.split("/")[1].split(".")[0] for i in glob.glob('LAB_NPY/*.npy')]
    for name in names:
        edf = np.load("EDF_NPY/" + name + ".npy")
        label = np.load("LAB_NPY/" + name + ".npy")
        row = [name, edf.shape, label.shape,
               np.count_nonzero(label[:, 0] == 0),
               np.count_nonzero(label[:, 0] == 1)]
        table.append(row)
    result = tabulate(table, headers, tablefmt="grid")
    print(result)



if __name__ == '__main__':
    dir_check()
    edf_to_npy()
    label_to_npy()
    show_data_info()