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

# summary.csv
# Model  Ave.AUC  Ave.F1  CV1.Ave.AUC  CV1.F1.AUC   ...    CVN.F1.AUC
#   A
#   B
#   C

INDEX = ["\\multirow{5}{*}{1st} & 1/5",
         " & 2/5", " & 3/5", " & 4/5", " & 5/5",
         "\\multirow{5}{*}{2nd} & 1/5",
         " & 2/5", " & 3/5", " & 4/5", " & 5/5",
         "\\multirow{5}{*}{3rd} & 1/5",
         " & 2/5", " & 3/5", " & 4/5", " & 5/5",
         "\\multirow{5}{*}{4th} & 1/5",
         " & 2/5", " & 3/5", " & 4/5", " & 5/5",
         "\\multirow{5}{*}{5th} & 1/5",
         " & 2/5", " & 3/5", " & 4/5", " & 5/5"]

def get_summary_path(modelname):
    if not os.path.exists(modelname):
        raise FileNotFoundError("Directories are not found!")

    summary_path = sorted(glob.glob('{}/summary*.csv'.format(modelname),
                                      recursive=False))
    print("{} summary was found in {}".format(len(summary_path), modelname))

    return summary_path[0]


def cul_averages(summary_path, n_trial, n_cv):
    data = pd.read_csv(summary_path)
    auc_ave = data.mean(axis=0)[0]
    f1_ave = data.mean(axis=0)[1]
    auc_aves_cv = [data.iloc[(n_cv*i):(n_cv*(i+1))].mean(axis=0)[0] 
                   for i in range(n_trial)]
    f1_aves_cv = [data.iloc[(n_cv*i):(n_cv*(i+1))].mean(axis=0)[1]
                  for i in range(n_trial)]
    result = [auc_ave, f1_ave]*(n_trial+1)
    result[2::2] = auc_aves_cv
    result[3::2] = f1_aves_cv

    return result


def make_multi_summary(modelnames, show_all=True):

    dt_now = datetime.datetime.now()

    if show_all:
        auc_df, f1_df = generate_auc_f1_df(modelnames)
        auc_df.to_csv('summary_{:%H%M%S}_auc.csv'.format(dt_now))
        f1_df.to_csv('summary_{:%H%M%S}_f1.csv'.format(dt_now))
    else:
        # create dataframe using info of the 1st called model
        summary_path_1st =  get_summary_path(modelnames[0])

        print(pd.read_csv(summary_path_1st).iloc[-1][0][4])
        n_trial = int(pd.read_csv(summary_path_1st).iloc[-1][0][2]) + 1
        
        n_cv = int(pd.read_csv(summary_path_1st).iloc[-1][0][4]) + 1
        n_models = len(modelnames)
        output_row = n_models
        output_col = (n_trial +1) * 2
        columns = ["Ave.AUC", "Ave.F1"]
        for i in range(n_trial):
            columns.append("CV{}_Ave.AUC".format(i))
            columns.append("CV{}_Ave.F1".format(i))
        print(columns)

        df = pd.DataFrame(np.zeros((output_row, output_col)), 
                        index=modelnames, columns=columns,
                        dtype=float)

        for name in modelnames:
            sum_path = get_summary_path(name)
            row = cul_averages(sum_path, n_trial, n_cv)
            print(row)
            df.loc[name] = row

        df.to_csv('summary_{:%H%M%S}.csv'.format(dt_now))
    

def generate_auc_f1_df(modelnames):
    auc_list =[]
    f1_list = []

    for modelname in modelnames:
        df = pd.read_csv(get_summary_path(modelname))
        auc_list.append(df["AUC"].tolist())
        f1_list.append(df["F1"].tolist())
    
    casename_list = pd.read_csv(get_summary_path(modelname))["Data"].tolist()
    print(casename_list)
    auc_df = pd.DataFrame(auc_list, index=modelnames, columns=casename_list).T
    f1_df = pd.DataFrame(f1_list, index=modelnames, columns=casename_list).T
    print(auc_df)
    print(f1_df)

    return auc_df, f1_df


def make_boxplot(modelnames):

    auc_df, f1_df = generate_auc_f1_df(modelnames)

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title("AUC", fontsize=24)
    ax2.set_title("F1", fontsize=24)
    #ax1.set_ylabel("AUC_value", fontsize=24)
    #ax2.set_ylabel("F1_value", fontsize=24)
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


def summary_to_tex(modelnames):

    auc_df, f1_df = generate_auc_f1_df(modelnames)

    save_path = 'summary_{:%H%M%S}.tex'.format(datetime.datetime.now())
    columnnames = [auc_df.columns[i//2] for i in range(len(auc_df.columns)*2)]
    print(columnnames)
    
    df = pd.DataFrame(np.zeros((len(auc_df.index), len(columnnames))),
                      index=auc_df.index, columns=columnnames)
    df.index = INDEX
    print(auc_df.loc[:, columnnames[0]])
    print(f1_df)
    for mi, modelname in enumerate(auc_df.columns):
        df.iloc[:, mi*2] = auc_df.loc[:, modelname]
        df.iloc[:, mi*2+1] = f1_df.loc[:, modelname]
    data = df.to_latex(float_format="%.3f")
    data = data.replace('{}', '\multicolumn{2}{c|}{Trial}')
    data = data.replace('\\textbackslash ', '\\').replace('\&', '&')
    data = data.replace('\{', '{').replace('\}', '}')
    data = data.replace("lrrrrr", "l|l|rrrrr")

    with open(save_path, mode='w') as (f):
        f.write('% Data \n')
        f.write(data)
        f.write('\n \n \n')

if __name__ == "__main__":
    
    print(len(sys.argv))
    modelnames = sys.argv[1:]
    pd.options.display.float_format = '{:.3g}'.format

    if len(sys.argv) < 3:
        print("Please choose at least 2 models!")
    else:
        make_multi_summary(modelnames, show_all=True)
        make_boxplot(modelnames)
        summary_to_tex(modelnames)