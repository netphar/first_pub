# zagidull@lm8-945-003:~/Documents/fimm_files/chemical_similarity/classifier$ scp -P 22 drugs_synergy_css_09012019.csv bzagidul@atlas.fimm.fi:/homes/bzagidul/test_grun/linR
# bzagidul@atlas.fimm.fi's password:
# /etc/profile.d/lang.sh: line 19: warning: setlocale: LC_CTYPE: cannot change locale (UTF-8): No such file or directory
# drugs_synergy_css_09012019.csv                                                          100%   43MB  10.7MB/s   00:04
# zagidull@lm8-945-003:~/Documents/fimm_files/chemical_similarity/classifier$ scp -P 22 summary_css_dss_ic50_synergy_smiles.csv bzagidul@atlas.fimm.fi:/homes/bzagidul/test_grun/linR
# scp -P 22 averages_17012019/* bzagidul@atlas.fimm.fi:/homes/bzagidul/test_grun/linR/averages_17012019/
# this is for running on atlas
#%%
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

#%%
# this retarded way is something we are doing to match filenames to cell lines
directory = os.fsencode('/homes/bzagidul/test_grun/linR/averages_17012019/')

count = 0
urgh = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        count = count + 1
        urgh.append(filename)
#        print(filename)
        continue
    else:
        continue
print(count)

#%%
urgh_cleaned = {}
for file in urgh:
    s = file
    match = re.findall("_(.*?)_", s)
    urgh_cleaned[s] = match[0]
#    print(match[0])
print(len(urgh_cleaned))

#%%
def sorter(x, name):
    out = x.loc[x['cell_line_name'] == name]
    out = out.loc[out['drug_row'] != 'ARSENIC TRIOXIDE']
    out = out.loc[out['drug_col'] != 'ARSENIC TRIOXIDE']
    return out


# r2_score(y_true, y_pred) # Correct!
# r2_score(y_pred, y_true) # Incorrect!!!!

def calc_train_R2(X_train, y_train, model):
    '''returns in-sample R2 for already fit model.'''
    predictions = model.predict(X_train)
    r2 = r2_score(y_train, predictions)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return r2, rmse


def calc_validation_R2(X_validation, y_validation, model):
    '''returns out-of-sample R2 for already fit model.'''
    predictions = model.predict(X_validation)
    r2 = r2_score(y_validation, predictions)
    mse = mean_squared_error(y_validation, predictions)
    rmse = np.sqrt(mse)
    return r2, rmse


def calc_metricsR2(X_train, y_train, X_validation, y_validation, model):
    '''fits model and returns the R2 for in-sample error and out-of-sample error'''
    model.fit(X_train, y_train)
    train_error, train_error_rmse = calc_train_R2(X_train, y_train, model)
    validation_error, validation_error_rmse = calc_validation_R2(X_validation, y_validation, model)
    return train_error, validation_error, train_error_rmse, validation_error_rmse



#%%
values_synergy_new = pd.read_csv('/homes/bzagidul/test_grun/linR/drugs_synergy_css_09012019.csv', sep=',')
names = np.unique(values_synergy_new['cell_line_name'])

smiles_holder = pd.read_csv('/homes/bzagidul/test_grun/linR/summary_css_dss_ic50_synergy_smiles.csv', sep=';')
smiles_holder.drop(columns=['Unnamed: 0'], inplace=True)
all_holder = {name: sorter(smiles_holder, name) for name in names}

#%%
os.chdir('/homes/bzagidul/test_grun/linR/averages_17012019')

to_drop = ['drug_row', 'drug_col', 'block_id', 'drug_row_id', 'drug_col_id', 'cell_line_id', 'synergy_zip',
           'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'ic50_row', 'ic50_col', 'cell_line_name', 'smiles_row',
           'smiles_col']

dropped_rows_all = {}
lr_r2_all = {}
lr_rmse_all = {}

for i, v in urgh_cleaned.items():
    test = pd.read_csv(i, sep=';')
    print(i,v)
    if v == 'NCI':
        v = 'NCI/ADR-RES'
    v = pd.merge(left=test, right=all_holder[v], left_on=['Drug1', 'Drug2'], right_on=['drug_row', 'drug_col'],
                 how='inner')
    v.drop(columns=['drug_row', 'drug_col'], inplace=True)
    v.rename(columns={'Drug1': 'drug_row', 'Drug2': 'drug_col'}, inplace=True)
    v.drop(to_drop, axis=1, inplace=True)
    target = v.pop('css')
    dt, tt = v, target

    # folds for inner CV
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    # folds for outer CV
    kfold_outer = KFold(n_splits=5, shuffle=True, random_state=2)

    # fit on split train data
    lr_split = LinearRegression(fit_intercept=False, n_jobs=-1)

    # placeholders for results
    r2_all_training, r2_all_validation, r2_all_test = [], [], []
    r2_naive_all_training, r2_naive_all_validation, r2_naive_all_test = [], [], []

    rmse_all_training, rmse_all_validation, rmse_all_test = [], [], []
    rmse_naive_all_training, rmse_naive_all_validation, rmse_naive_all_test = [], [], []
    holder_outer = {}
    holder_inner = {}
    # enumerate splits
    iteration_outer = 0
    iteration_inner = 0
    for test, test_val in kfold_outer.split(dt):  # doing the split

        xt, yt = dt.iloc[test], tt.iloc[test]
        xtt, ytt = dt.iloc[test_val], tt.iloc[test_val]

        # storing data
        tostore = dict(zip(['test', 'test_val', ], [test, test_val]))
        holder_outer[iteration_outer] = tostore

        for train, validation in kfold.split(xt):
            # leaving css_row and css_col increases r2 by 0.1
            # so we gotta pop them off
            xt1 = xt.copy()
            xtt1 = xtt.copy()
            xt2 = xt.copy()
            [xt2.pop(x) for x in ['dss_row', 'dss_col']]
            xtt2 = xtt.copy()
            [xtt2.pop(x) for x in ['dss_row', 'dss_col']]

            naive_training = pd.concat([xt1.pop(x) for x in ['dss_row', 'dss_col']], 1)
            #    naive_training['averages'] = naive_training.mean(numeric_only=True, axis=1) # css = (dss1+dss2)/2
            naive_training['sum'] = naive_training.sum(numeric_only=True, axis=1)  # css = dss1 + dss2
            #    naive_training = naive_training.pop('averages')
            naive_training = naive_training.pop('sum')

            naive_test = pd.concat([xtt1.pop(x) for x in ['dss_row', 'dss_col']], 1)
            #    naive_test['averages'] = naive_test.mean(numeric_only=True, axis=1) # css = (dss1+dss2)/2
            naive_test['sum'] = naive_test.sum(numeric_only=True, axis=1)  # css = dss1 + dss2

            #    naive_test = naive_test.pop('averages')
            naive_test = naive_test.pop('sum')

            naive_r2_training = r2_score(yt.iloc[train], naive_training.iloc[train])
            naive_r2_validation = r2_score(yt.iloc[validation], naive_training.iloc[validation])
            naive_r2_test = r2_score(ytt, naive_test)

            naive_rmse_training = np.sqrt(mean_squared_error(yt.iloc[train], naive_training.iloc[train]))
            naive_rmse_validation = np.sqrt(mean_squared_error(yt.iloc[validation], naive_training.iloc[validation]))
            naive_rmse_test = np.sqrt(mean_squared_error(ytt, naive_test))

            # this is R2 for linear regression, inner CV
            lr_split.fit(xt2.iloc[train], yt.iloc[train])
            r2_training, r2_validation, rmse_training, rmse_validation = calc_metricsR2(xt2.iloc[train], yt.iloc[train],
                                                                                        xt2.iloc[validation],
                                                                                        yt.iloc[validation], lr_split)
            yt_predicted = lr_split.predict(xtt2)
            r2_test = r2_score(ytt, yt_predicted)
            rmse_test = np.sqrt(mean_squared_error(ytt, yt_predicted))
            tostore1 = dict(zip(['train', 'validation', 'xt2.iloc[train]', 'yt.iloc[train]', 'xt2.iloc[validation]',
                                 'yt.iloc[validation]', 'ytt', 'yt_predicted', 'xtt'],
                                [train, validation, xt2.iloc[train], yt.iloc[train], xt2.iloc[validation],
                                 yt.iloc[validation], ytt, yt_predicted, xtt]))
            holder_inner[iteration_inner] = tostore1

            # append to result holders
            r2_training, r2_validation, r2_test = round(r2_training, 3), round(r2_validation, 3), round(r2_test, 3)
            naive_r2_training, naive_r2_validation, naive_r2_test = round(naive_r2_training, 3), round(
                naive_r2_validation, 3), round(naive_r2_test, 3)

            rmse_training, rmse_validation, rmse_test = round(rmse_training, 3), round(rmse_validation, 3), round(
                rmse_test, 3)
            naive_rmse_training, naive_rmse_validation, naive_rmse_test = round(naive_rmse_training, 3), round(
                naive_rmse_validation, 3), round(naive_rmse_test, 3)

            r2_all_training.append(r2_training)
            r2_all_validation.append(r2_validation)
            r2_all_test.append(r2_test)

            r2_naive_all_training.append(naive_r2_training)
            r2_naive_all_validation.append(naive_r2_validation)
            r2_naive_all_test.append(naive_r2_test)

            rmse_all_training.append(rmse_training)
            rmse_all_validation.append(rmse_validation)
            rmse_all_test.append(rmse_test)

            rmse_naive_all_training.append(naive_rmse_training)
            rmse_naive_all_validation.append(naive_rmse_validation)
            rmse_naive_all_test.append(naive_rmse_test)

            print('---')
            print('cell line: %s' % i)
            print('outer iteration id: %s, inner iteration id: %s' % (iteration_outer, iteration_inner))
            print('r2_training: %s, r2_validation: %s, r2_test: %s' % (r2_training, r2_validation, r2_test))
            print('naive_r2_training: %s, naive_r2_validation: %s, naive_r2_test: %s' % (naive_r2_training, naive_r2_validation, naive_r2_test))
            print('rmse_training: %s, rmse_validation: %s, rmse_test: %s' % (rmse_training, rmse_validation, rmse_test))
            print('naive_rmse_training: %s, naive_rmse_validation: %s, naive_rmse_test: %s' % (naive_rmse_training, naive_rmse_validation, naive_rmse_test))
            iteration_inner = iteration_inner + 1
        iteration_outer = iteration_outer + 1

    linR_r2_fitF = pd.concat([pd.Series(x) for x in
                              [r2_all_training, r2_naive_all_training, r2_all_validation, r2_naive_all_validation,
                               r2_all_test, r2_naive_all_test]], 1)
    linR_rmse_fitF = pd.concat([pd.Series(x) for x in [rmse_all_training, rmse_naive_all_training, rmse_all_validation,
                                                       rmse_naive_all_validation, rmse_all_test, rmse_naive_all_test]],
                               1)

    linR_r2_fitF.rename(
        columns={0: "LR_training", 1: "Naive_training", 2: "LR_validation", 3: "Naive_validation", 4: "LR_test",
                 5: "Naive_test"}, inplace=True)
    linR_rmse_fitF.rename(
        columns={0: "LR_training", 1: "Naive_training", 2: "LR_validation", 3: "Naive_validation", 4: "LR_test",
                 5: "Naive_test"}, inplace=True)

    # dropping df = df.drop(df[(df.score < 50) & (df.score > 20)].index)
    error_rr = linR_r2_fitF[(linR_r2_fitF.LR_test > 1) | (linR_r2_fitF.LR_test < 0)].index
    dropped_rows_all[i] = error_rr
    print('dropped rows: %s' % error_rr)

    linR_r2_fitF = linR_r2_fitF.drop(error_rr)
    lr_r2_all[i] = linR_r2_fitF
    linR_rmse_fitF = linR_rmse_fitF.drop(error_rr)
    lr_rmse_all[i] = linR_rmse_fitF

with open('lr_r2_all.pickle', 'wb') as f:
    pickle.dump(lr_r2_all, f, pickle.HIGHEST_PROTOCOL)
with open('lr_rmse_all.pickle', 'wb') as f:
    pickle.dump(lr_rmse_all, f, pickle.HIGHEST_PROTOCOL)
