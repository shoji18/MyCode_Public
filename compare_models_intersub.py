import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys
import datetime
import scipy
from scikit_posthocs import sign_array
from scikit_posthocs import posthoc_nemenyi_friedman as nemenyi_test
import seaborn as sns

# for get 1 model result
def make_single_summary(dirname=""):

    if dirname == "":
        folders_cd = glob.glob("**/")
        print("Results:")
        for i, dirname in enumerate(folders_cd):
            print("{}: {}".format(i, dirname))
        n_chosen = int(input("Please choose a dir number: "))
        dirname = folders_cd[n_chosen]

    summary_paths =  sorted(glob.glob('{}/**/summary*.csv'.format(dirname),
                                      recursive=True))
    name_list = sorted([path.split('/')[2].split('_')[1].split('.')[0] \
                        for path in summary_paths])
    metrics = pd.read_csv(summary_paths[0]).columns.tolist()[1:]

    len_row = len(name_list)
    len_col = len(metrics)

    df = pd.DataFrame(np.zeros((len_row, len_col), dtype=float),
                                columns=metrics,
                                index=name_list)
    df = df.astype({'TN': int, 'FP': int, 'FN': int, 'TP': int})

    for i in range(len(summary_paths)):
        data = pd.read_csv(summary_paths[i]).iloc[0][1:]
        df.iloc[i] = data.tolist()

    df.to_csv('{}/summary_concat.csv'.format(dirname))


def get_summary_paths(modelname):
    if not os.path.exists(modelname):
        raise FileNotFoundError("Directories are not found!")

    summary_paths = sorted(glob.glob('{}/**/summary*.csv'.format(modelname),
                                      recursive=False))
    print("{} summaries were found in {}".format(len(summary_paths), modelname))

    return summary_paths


def make_multi_summary(modelname_list):

    # create dataframe using info of the 1st called model
    summary_paths_1st =  get_summary_paths(modelname_list[0])
    metrics = pd.read_csv(summary_paths_1st[0]).columns.tolist()[1:]

    n_cases = len(summary_paths_1st)
    n_models = len(modelname_list)
    output_row = n_cases * n_models
    output_col = len(metrics)

    df = pd.DataFrame(np.zeros((output_row, output_col), dtype=float))
    df.columns = metrics
    df = df.astype({'TN': int, 'FP': int, 'FN': int, 'TP': int})

    name_list = sorted([path.split('/')[2].split('_')[1].split('.')[0]
                        for path in summary_paths_1st])

    # read data and store in df
    summary_path_all = [get_summary_paths(i) for i in modelname_list]

    for mi in range(n_models):
        for ci in range(n_cases):
            data = pd.read_csv(summary_path_all[mi][ci]).iloc[0][1:]
            df = df.rename(index = {mi + n_models*ci:'{}_+x=x+_{}'
                                    .format(modelname_list[mi], name_list[ci])})
            df.iloc[mi + n_models*ci] = data.tolist()

    df.to_csv('summary_{:%H%M%S}.csv'.format(datetime.datetime.now()))

    make_boxplot(modelname_list, df)


def integrate_results(command_args):
    if len(command_args) == 1:
        make_single_summary()
    elif len(command_args) == 2:
        make_single_summary(command_args[1])
    else:
        make_multi_summary(command_args[1:])


def validation_check(error, modelnames):

    print("IndexError")
    print(error)
    print("Validation checking...")
    
    for modelname in modelnames[1:]:
        summary_paths =  get_summary_paths(modelname)
        for sum_path in summary_paths:
            data = pd.read_csv(sum_path)
            if len(data)<1:
                print("Empty data detected!")
                print("path: {}".format(sum_path))
                print("data: {}".format(data))


def make_boxplot(modelnames, dataframe):
    print(dataframe)
    auc_list =[]
    f1_list = []
    casename_list = []
    n_models = len(modelnames)
    n_cases = len(dataframe)//n_models

    for idx, modelname in enumerate(modelnames):
        auc_list.append([dataframe.iloc[i*n_models+idx, 0] for i in range(n_cases)])
        f1_list.append([dataframe.iloc[i*n_models+idx, 1] for i in range(n_cases)])
    
    casename_list = [i.split("_+x=x+_")[-1] for i 
                     in dataframe.index.values[::n_models]]
    print("casename", casename_list)
    auc_df = pd.DataFrame(auc_list, index=modelnames, columns=casename_list)
    f1_df = pd.DataFrame(f1_list, index=modelnames, columns=casename_list)

    nan_check_list = pd.isnull(auc_df).any(0).tolist()
    nan_index_list = [i for i, x in enumerate(nan_check_list) if x == True]
    
    auc_df = auc_df.drop(auc_df.columns[nan_index_list], axis=1).T
    f1_df = f1_df.drop(f1_df.columns[nan_index_list], axis=1).T
    
    aucs = [auc_df[name].values.tolist() for name in modelnames]
    f1s = [f1_df[name].values.tolist() for name in modelnames]

    stat_auc, p_value_auc \
        = scipy.stats.friedmanchisquare(*aucs)
    stat_f1, p_value_f1 \
        = scipy.stats.friedmanchisquare(*f1s)
    nemenyi_result_auc \
        = nemenyi_test(np.array(aucs).transpose(1,0))
    nemenyi_result_f1 \
        = nemenyi_test(np.array(f1s).transpose(1,0))
    
    sign_auc = sign_array(nemenyi_result_auc)
    sign_f1 = sign_array(nemenyi_result_f1)

    print("Freidman test result(AUC):", stat_auc)
    print("Freidman test p-value(AUC):", p_value_auc)
    print("Freidman test result(F1):", stat_f1)
    print("Freidman test p-value(F1):", p_value_f1)
    print("nemenyi result(AUC):", nemenyi_result_auc)
    print("nemenyi result(F1):", nemenyi_result_f1)

    significance_list_auc = []
    significance_list_f1 = []

    for col in range(sign_auc.shape[1]-1):
        if sign_auc[-1, col]==1:
            significance_list_auc.append([sign_auc.shape[0]-1, col])
        if sign_f1[-1, col]==1:
            significance_list_f1.append([sign_auc.shape[0]-1, col])

    print(significance_list_auc)
    print(significance_list_f1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.set_title("AUC", fontsize=24)
    ax2.set_title("F1", fontsize=24)
    #ax1.set_ylabel("AUC_value")
    #ax2.set_ylabel("F1_value")
    ax1.set_yticks([0, 0.3, 0.5, 0.8, 1.0])
    ax2.set_yticks([0, 0.3, 0.5, 0.8, 1.0])
    ax1.set_yticklabels([0, 0.3, 0.5, 0.8, 1.0], fontsize=14)
    ax2.set_yticklabels([0, 0.3, 0.5, 0.8, 1.0], fontsize=14)
    ax1.set_ylim([0, 1.15])
    ax2.set_ylim([0, 1.15])
    ax1.set_xticklabels(modelnames, rotation=45, ha='right', fontsize=14)
    ax2.set_xticklabels(modelnames, rotation=45, ha='right', fontsize=14)
    sns.boxplot(data=auc_df, ax=ax1)
    sns.swarmplot(data=auc_df, ax=ax1, color="gray")
    for i in range(len(significance_list_auc)):
        x1, x2 = significance_list_auc[i]
        y, h = 1 + i*0.03 +0.03, 0.01
        ax1.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black")
        ax1.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom')
    sns.boxplot(data=f1_df, ax=ax2)
    sns.swarmplot(data=f1_df, ax=ax2, color="gray")
    for i in range(len(significance_list_f1)):
        x1, x2 = significance_list_f1[i]
        y, h = 1 + i*0.03 +0.03, 0.01
        ax2.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color="black")
        ax2.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom')
    plt.savefig("boxplot_{:%H%M%S}.pdf".format(datetime.datetime.now()))


if __name__ == "__main__":
    
    command_args = sys.argv
    print(len(command_args))

    try:
        integrate_results(command_args)
    except IndexError as error:
        validation_check(error, command_args)