'''
THIS VERSION IS USED FOR RUNNING ON DrugComb server

AGAIN FOR RUNNING ON ATLAS
https://stats.stackexchange.com/questions/139042/ensemble-of-different-kinds-of-regressors-using-scikit-learn-or-any-other-pytho
zagidull@lm8-945-003:~$ scp -P 22 /Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drugs_synergy_css_09012019_updatedCSS.csv bzagidul@atlas.fimm.fi:/homes/bzagidul/test_grun/linR_svr_kr
zagidull@lm8-945-003:~$ scp -P 22 /Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/summary_css_dss_ic50_synergy_smiles.csv bzagidul@atlas.fimm.fi:/homes/bzagidul/test_grun/linR_svr_kr
scp -P 22 /Users/zagidull/PycharmProjects/test_scientific/case_3_ensemble.py bzagidul@atlas.fimm.fi:/homes/bzagidul/test_grun/linR_svr_kr
'''

import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# %%
# this retarded way is something we are doing to match filenames to cell lines
directory = os.fsencode('/home/bulat/KBM7/averages_17012019')

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

# %%
urgh_cleaned = {}
for file in urgh:
    s = file
    match = re.findall("_(.*?)_", s)
    urgh_cleaned[s] = match[0]
#    print(match[0])
print(len(urgh_cleaned))


# %%
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


# %%
values_synergy_new = pd.read_csv(
    '/home/bulat/KBM7/drugs_synergy_css_09012019_updatedCSS.csv',
    sep=';')
names = np.unique(values_synergy_new['cell_line_name'])

smiles_holder = pd.read_csv(
    '/home/bulat/KBM7/summary_css_dss_ic50_synergy_smiles.csv',
    sep=';')
smiles_holder.drop(columns=['Unnamed: 0'], inplace=True)
all_holder = {name: sorter(smiles_holder, name) for name in names}

# %%
os.chdir('/home/bulat/KBM7/averages_17012019')

to_drop = ['drug_row', 'drug_col', 'block_id', 'drug_row_id', 'drug_col_id', 'cell_line_id', 'synergy_zip',
           'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'ic50_row', 'ic50_col', 'cell_line_name', 'smiles_row',
           'smiles_col']

dropped_rows_all = {}
kr_rmse_all = {}
lr_rmse_all = {}
svr_rmse_all = {}
ridge_rmse_all = {}
lasso_rmse_all = {}

for i, v in urgh_cleaned.items():
    if v == 'KBM-7':
        test = pd.read_csv(i, sep=';')
        print(i, v)
        v = pd.merge(left=test, right=all_holder[v], left_on=['Drug1', 'Drug2'], right_on=['drug_row', 'drug_col'],
                     how='inner')
        v.drop(columns=['drug_row', 'drug_col'], inplace=True)
        v.rename(columns={'Drug1': 'drug_row', 'Drug2': 'drug_col'}, inplace=True)
        v.drop(to_drop, axis=1, inplace=True)
        target = v.pop('css')
        dt, tt = v, target

        # folds for inner CV
        kfold = KFold(n_splits=5, shuffle=True, random_state=1)

        # folds for outer CV
        kfold_outer = KFold(n_splits=2, shuffle=True, random_state=2)

        # fit on split train data
        lr_split = LinearRegression(fit_intercept=False, n_jobs=-1)
        svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                           param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                       "gamma": np.logspace(-2, 2, 5)}, n_jobs=-1)

        kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), cv=5,
                          param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                                      "gamma": np.logspace(-2, 2, 5)}, n_jobs=-1)
        ridge = GridSearchCV(Ridge(), cv=5, param_grid={"alpha": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],
                                                        "fit_intercept": ['True', 'False']}, n_jobs=-1)
        lasso = GridSearchCV(Lasso(), cv=5, param_grid={"alpha": [0.01, 0.1, 1, 10, 100],
                                                        "fit_intercept": ['True', 'False']}, n_jobs=-1)

        # placeholders for results
        rmse_all_training_svr, rmse_all_validation_svr, rmse_all_test_svr = [], [], []
        rmse_all_training_kr, rmse_all_validation_kr, rmse_all_test_kr = [], [], []

        rmse_all_training_ridge, rmse_all_validation_ridge, rmse_all_test_ridge = [], [], []
        rmse_all_training_lasso, rmse_all_validation_lasso, rmse_all_test_lasso = [], [], []

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

                naive_rmse_training = np.sqrt(mean_squared_error(yt.iloc[train], naive_training.iloc[train]))
                naive_rmse_validation = np.sqrt(
                    mean_squared_error(yt.iloc[validation], naive_training.iloc[validation]))
                naive_rmse_test = np.sqrt(mean_squared_error(ytt, naive_test))

                # this is R2 for linear regression, inner CV
                lr_split.fit(xt2.iloc[train], yt.iloc[train])
                svr.fit(xt2.iloc[train], yt.iloc[train])
                kr.fit(xt2.iloc[train], yt.iloc[train])
                lasso.fit(xt2.iloc[train], yt.iloc[train])
                ridge.fit(xt2.iloc[train], yt.iloc[train])

                _, _, rmse_training, rmse_validation = calc_metricsR2(xt2.iloc[train], yt.iloc[train],
                                                                      xt2.iloc[validation],
                                                                      yt.iloc[validation], lr_split)
                _, _, rmse_training_svr, rmse_validation_svr = calc_metricsR2(xt2.iloc[train], yt.iloc[train],
                                                                              xt2.iloc[validation],
                                                                              yt.iloc[validation], svr)
                _, _, rmse_training_kr, rmse_validation_kr = calc_metricsR2(xt2.iloc[train], yt.iloc[train],
                                                                            xt2.iloc[validation],
                                                                            yt.iloc[validation], kr)
                _, _, rmse_training_lasso, rmse_validation_lasso = calc_metricsR2(xt2.iloc[train], yt.iloc[train],
                                                                            xt2.iloc[validation],
                                                                            yt.iloc[validation], lasso)
                _, _, rmse_training_ridge, rmse_validation_ridge = calc_metricsR2(xt2.iloc[train], yt.iloc[train],
                                                                            xt2.iloc[validation],
                                                                            yt.iloc[validation], ridge)
                yt_predicted = lr_split.predict(xtt2)
                yt_predicted_svr = svr.predict(xtt2)
                yt_predicted_kr = kr.predict(xtt2)
                yt_predicted_lasso = lasso.predict(xtt2)
                yt_predicted_ridge = ridge.predict(xtt2)

                rmse_test = np.sqrt(mean_squared_error(ytt, yt_predicted))
                rmse_test_svr = np.sqrt(mean_squared_error(ytt, yt_predicted_svr))
                rmse_test_kr = np.sqrt(mean_squared_error(ytt, yt_predicted_kr))
                rmse_test_ridge = np.sqrt(mean_squared_error(ytt, yt_predicted_lasso))
                rmse_test_lasso = np.sqrt(mean_squared_error(ytt, yt_predicted_ridge))

                tostore1 = dict(zip(['train', 'validation', 'xt2.iloc[train]', 'yt.iloc[train]', 'xt2.iloc[validation]',
                                     'yt.iloc[validation]', 'ytt', 'yt_predicted', 'yt_predicted_svr',
                                     'yt_predicted_kr', 'yt_predicted_lasso', 'yt_predicted_ridge', 'xtt'],
                                    [train, validation, xt2.iloc[train], yt.iloc[train], xt2.iloc[validation],
                                     yt.iloc[validation], ytt, yt_predicted, yt_predicted_svr, yt_predicted_kr,
                                     yt_predicted_lasso, yt_predicted_ridge, xtt]))
                holder_inner[iteration_inner] = tostore1

                # append to result holders
                rmse_training, rmse_validation, rmse_test = round(rmse_training, 3), round(rmse_validation, 3), round(
                    rmse_test, 3)
                rmse_training_kr, rmse_validation_kr, rmse_test_kr = round(rmse_training_kr, 3), round(
                    rmse_validation_kr, 3), round(
                    rmse_test_kr, 3)
                rmse_training_svr, rmse_validation_svr, rmse_test_svr = round(rmse_training_svr, 3), round(
                    rmse_validation_svr, 3), round(
                    rmse_test_svr, 3)
                rmse_training_lasso, rmse_validation_lasso, rmse_test_lasso = round(rmse_training_lasso, 3), round(
                    rmse_validation_lasso, 3), round(
                    rmse_test_lasso, 3)
                rmse_training_ridge, rmse_validation_ridge, rmse_test_ridge = round(rmse_training_ridge, 3), round(
                    rmse_validation_ridge, 3), round(
                    rmse_test_ridge, 3)
                naive_rmse_training, naive_rmse_validation, naive_rmse_test = round(naive_rmse_training, 3), round(
                    naive_rmse_validation, 3), round(naive_rmse_test, 3)

                rmse_all_training.append(rmse_training)
                rmse_all_validation.append(rmse_validation)
                rmse_all_test.append(rmse_test)

                rmse_all_training_kr.append(rmse_training_kr)
                rmse_all_validation_kr.append(rmse_validation_kr)
                rmse_all_test_kr.append(rmse_test_kr)

                rmse_all_training_lasso.append(rmse_training_lasso)
                rmse_all_validation_lasso.append(rmse_validation_lasso)
                rmse_all_test_lasso.append(rmse_test_lasso)

                rmse_all_training_ridge.append(rmse_training_ridge)
                rmse_all_validation_ridge.append(rmse_validation_ridge)
                rmse_all_test_ridge.append(rmse_test_ridge)

                rmse_all_training_svr.append(rmse_training_svr)
                rmse_all_validation_svr.append(rmse_validation_svr)
                rmse_all_test_svr.append(rmse_test_svr)

                rmse_naive_all_training.append(naive_rmse_training)
                rmse_naive_all_validation.append(naive_rmse_validation)
                rmse_naive_all_test.append(naive_rmse_test)

                print('---')
                print('cell line: %s' % i)
                print('outer iteration id: %s, inner iteration id: %s' % (iteration_outer, iteration_inner))
                print('rmse_training: %s, rmse_validation: %s, rmse_test: %s' % (
                rmse_training, rmse_validation, rmse_test))
                print('rmse_training_kr: %s, rmse_validation_kr: %s, rmse_test_kr: %s' % (
                rmse_training_kr, rmse_validation_kr, rmse_test_kr))
                print('rmse_training_lasso: %s, rmse_validation_lasso: %s, rmse_test_lasso: %s' % (
                rmse_training_lasso, rmse_validation_lasso, rmse_test_lasso))
                print('rmse_training_ridge: %s, rmse_validation_ridge: %s, rmse_test_ridge: %s' % (
                rmse_training_ridge, rmse_validation_ridge, rmse_test_ridge))
                print('rmse_training_svr: %s, rmse_validation_svr: %s, rmse_test_svr: %s' % (
                rmse_training_svr, rmse_validation_svr, rmse_test_svr))
                print('naive_rmse_training: %s, naive_rmse_validation: %s, naive_rmse_test: %s' % (
                naive_rmse_training, naive_rmse_validation, naive_rmse_test))
                iteration_inner = iteration_inner + 1
            iteration_outer = iteration_outer + 1

        linR_rmse_fitF = pd.concat(
            [pd.Series(x) for x in [rmse_all_training, rmse_naive_all_training, rmse_all_validation,
                                    rmse_naive_all_validation, rmse_all_test, rmse_naive_all_test]],
            1)
        linR_rmse_fitF_svr = pd.concat(
            [pd.Series(x) for x in [rmse_all_training_svr, rmse_naive_all_training, rmse_all_validation_svr,
                                    rmse_naive_all_validation, rmse_all_test_svr, rmse_naive_all_test]],
            1)
        linR_rmse_fitF_kr = pd.concat(
            [pd.Series(x) for x in [rmse_all_training_kr, rmse_naive_all_training, rmse_all_validation_kr,
                                    rmse_naive_all_validation, rmse_all_test_kr, rmse_naive_all_test]],
            1)
        linR_rmse_fitF_ridge = pd.concat(
            [pd.Series(x) for x in [rmse_all_training_ridge, rmse_naive_all_training, rmse_all_validation_ridge,
                                    rmse_naive_all_validation, rmse_all_test_ridge, rmse_naive_all_test]],
            1)
        linR_rmse_fitF_lasso = pd.concat(
            [pd.Series(x) for x in [rmse_all_training_lasso, rmse_naive_all_training, rmse_all_validation_lasso,
                                    rmse_naive_all_validation, rmse_all_test_lasso, rmse_naive_all_test]],
            1)

        linR_rmse_fitF.rename(
            columns={0: "LR_training", 1: "Naive_training", 2: "LR_validation", 3: "Naive_validation", 4: "LR_test",
                     5: "Naive_test"}, inplace=True)
        linR_rmse_fitF_svr.rename(
            columns={0: "SVR_training", 1: "Naive_training", 2: "SVR_validation", 3: "Naive_validation", 4: "SVR_test",
                     5: "Naive_test"}, inplace=True)
        linR_rmse_fitF_kr.rename(
            columns={0: "KR_training", 1: "Naive_training", 2: "KR_validation", 3: "Naive_validation", 4: "KR_test",
                     5: "Naive_test"}, inplace=True)
        linR_rmse_fitF_lasso.rename(
            columns={0: "Lasso_training", 1: "Naive_training", 2: "Lasso_validation", 3: "Naive_validation", 4: "Lasso_test",
                     5: "Naive_test"}, inplace=True)
        linR_rmse_fitF_ridge.rename(
            columns={0: "Ridge_training", 1: "Naive_training", 2: "Ridge_validation", 3: "Naive_validation", 4: "Ridge_test",
                     5: "Naive_test"}, inplace=True)

        lr_rmse_all[i] = linR_rmse_fitF
        kr_rmse_all[i] = linR_rmse_fitF_kr
        svr_rmse_all[i] = linR_rmse_fitF_svr
        ridge_rmse_all[i] = linR_rmse_fitF_ridge
        lasso_rmse_all[i] = linR_rmse_fitF_lasso

    else:
        pass

with open('/home/bulat/KBM7/lr_rmse_all.pickle', 'wb') as f:
    pickle.dump(lr_rmse_all, f, pickle.HIGHEST_PROTOCOL)
with open('/home/bulat/KBM7/kr_rmse_all.pickle', 'wb') as f:
    pickle.dump(kr_rmse_all, f, pickle.HIGHEST_PROTOCOL)
with open('/home/bulat/KBM7/svr_rmse_all.pickle', 'wb') as f:
    pickle.dump(svr_rmse_all, f, pickle.HIGHEST_PROTOCOL)
with open('/home/bulat/KBM7/lasso_rmse_all.pickle', 'wb') as f:
    pickle.dump(lasso_rmse_all, f, pickle.HIGHEST_PROTOCOL)
with open('/home/bulat/KBM7/ridge_rmse_all.pickle', 'wb') as f:
    pickle.dump(ridge_rmse_all, f, pickle.HIGHEST_PROTOCOL)
