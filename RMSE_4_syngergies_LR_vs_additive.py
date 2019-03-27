# %%
import numpy as np
import pandas as pd
import os,re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import datetime
import matplotlib.collections
import seaborn as sns
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
# below are settings for the sns library
sns.set(rc={'figure.figsize': (62.8, 62.8)}, color_codes=True)
sns.set_style("whitegrid")
sns.set_context('paper')

# %%
# here we get a list of csv files with bitwise averages

categ = '/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/'
directory = os.fsencode(categ + 'averages_17012019')

count = 0
list_csv_files = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        count = count + 1
        list_csv_files.append(filename)
        #        print(filename)
        continue
    else:
        continue
print('These files: {0}\ncontain {1} cell lines'.format(list_csv_files, count))

# %%
# here we get files (as keys) with corresponding cell line names (as values) in a dict
files_cellnames = {}
for file in list_csv_files:
    if file == '_NCI_ADR-RES_smiles_17012019.csv':  # because pattern matching gives us onle "NCI" for this cell line
        files_cellnames[file] = 'NCI/ADR-RES'
    else:
        match = re.findall("_(.*?)_", file)
        files_cellnames[file] = match[0]
print('Dictionary containing these key-value pairs: {0} holds {1} cell lines'.format(files_cellnames,
                                                                                     len(files_cellnames)))


# %%
def sorter(x, name):
    '''
    :param x: input df with all cell lines
    :param name: cell line name
    :return: returns df with summary info (for drugs), without ARSENIC TRIOXIDE, since for these SMILES generated are not considered to be valid
    '''
    out = x.loc[x['cell_line_name'] == name]
    out = out.loc[out['drug_row'] != 'ARSENIC TRIOXIDE']
    out = out.loc[out['drug_col'] != 'ARSENIC TRIOXIDE']
    return out


# %%
def calculate_synergy_null(drug, df):
    '''
    :param df: pandas dataframe containing synergy values and drug id's
    :param drug: id of drug to calculate average synergy of combinations containing this drug
    :return: pd.Series with drugs as keys and values : {dict containing 4 synergy values}
    '''
    var = df[(df.drug_row_id == drug) | (df.drug_col_id == drug)][
        ['synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa']].mean()
    #     dict(zip(drug_id))
    return var


def calc_train_RMSE(X_train, y_train, model):
    '''returns in-sample RMSE for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return rmse


def calc_validation_RMSE(X_validation, y_validation, model):
    '''returns out-of-sample RMSE for already fit model.'''
    predictions = model.predict(X_validation)
    mse = mean_squared_error(y_validation, predictions)
    rmse = np.sqrt(mse)
    return rmse

def calc_test_RMSE(X_test, y_test, model):
    '''returns out-of-sample RMSE for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse

def calc_metrics_RMSE(X_train, y_train, X_validation, y_validation, X_test, y_test, model):
    '''
    :param X_train:
    :param y_train:
    :param X_validation:
    :param y_validation:
    :param X_test:
    :param y_test:
    :param model:
    :return: returns RMSE error == -1, if the RMSE is found to be above some threshold (1000 in this case)
    '''
    model.fit(X_train, y_train)
    train_error_rmse = calc_train_RMSE(X_train, y_train, model)
    validation_error_rmse = calc_validation_RMSE(X_validation, y_validation, model)
    test_error_rmse = calc_test_RMSE(X_test, y_test, model)
    if any(t > 1000 for t in [train_error_rmse, validation_error_rmse, test_error_rmse]):
        train_error_rmse = -1
        test_error_rmse = -1
        validation_error_rmse = -1
    return round(train_error_rmse, 3), round(validation_error_rmse, 3), round(test_error_rmse, 3)


def compareLR(model, train_index, test_index, features, target, target_null, lol=False):
    '''
    :param model: sklearn initialized model
    :param train_index: kfold object
    :param test_index: kfold object
    :param features: df with bit-wise averages
    :param target: values to predict
    :param target_null: values from null model
    :return: LR RMSE as int, Naive RMSE as int
    '''
    model.fit(features.iloc[train_index], target.iloc[train_index])

    pred = model.predict(features.iloc[test_index])
    e1 = np.sqrt(mean_squared_error(target.iloc[test_index], pred))  # LR
    e2 = np.sqrt(mean_squared_error(target.iloc[test_index], target_null.iloc[test_index]))  # Naive
    # if (e1 > 100*e2):
    #     print('shit')
    #     x = pred.argmin()
    #     y = pred.argmax()
    #     target.iloc[test_index].pop(x)
    #     target.iloc[test_index].pop(y)
    #     pred.pop(y)
    #     pred.pop(x)
    #     print(np.sqrt(mean_squared_error(target.iloc[test_index], pred)))
    #     print(np.sqrt(mean_squared_error(target.iloc[test_index], target_null.iloc[test_index])))
    #
    #
    # if lol: # delete outliers?
    #     x = pred.argmin()
    #     pred = np.delete(pred, x)
    #     pred1 = target.iloc[test_index].drop(target.iloc[test_index].index[x])
    #     e1 = np.sqrt(mean_squared_error(pred1, pred))
    #
    #
    #     return e1, e2
    # else:
    return e1, e2




# %%
# this part is training
values_synergy_new = pd.read_csv(categ + 'drugs_synergy_css_09012019_updatedCSS.csv', sep=';')  # this file contains summary info without SMILES strings
names = np.unique(values_synergy_new['cell_line_name'])

smiles_holder = pd.read_csv(categ + 'summary_css_dss_ic50_synergy_smiles.csv', sep=';') #  this file contains summary info (synergies, css, dss, ic50, etc without SMILES strings
smiles_holder.drop(columns=['Unnamed: 0'], inplace=True)
all_holder = {name: sorter(smiles_holder, name) for name in names}  # all_holder is a dict which uses cell line names as keys and contains all summary data + SMILES strings

# %%
# here we create a training dataset with S_score and four synergy models as targets to predict
# further on Linear Regression iterates through the df's using 5-fold CV
# on each fold we test S_score and synergy values and getting their RMSE values
# only synergy scores have been used to make the violion plots. NOT the S_score

os.chdir(categ + 'averages_17012019')

to_drop = ['drug_row', 'drug_col', 'drug_row_id', 'drug_col_id', 'cell_line_id', 'css', 'ic50_row', 'ic50_col',
           'dss_row',
           'dss_col', 'cell_line_name', 'smiles_row', 'smiles_col',
           'row_zip_null', 'col_zip_null', 'row_bliss_null', 'col_bliss_null',
           'row_loewe_null', 'col_loewe_null', 'row_hsa_null', 'col_hsa_null']

to_drop_synergies = ['synergy_zip', 'synergy_bliss', 'synergy_loewe', 'synergy_hsa']

holder = {}  # holds all data

for key, value in files_cellnames.items():
    print(key)

#    if value in ['COLO 205']:
    test = pd.merge(left=pd.read_csv(key, sep=';'), right=all_holder[value], left_on=['Drug1', 'Drug2'],
                    right_on=['drug_row', 'drug_col'],
                    how='inner')
    test.drop(columns=['drug_row', 'drug_col'], inplace=True)
    test.rename(columns={'Drug1': 'drug_row', 'Drug2': 'drug_col'}, inplace=True)

    # getting S_score
    test['S_score_real'] = test['css'] - test[['dss_row', 'dss_col']].max(axis=1)
    test['S_score_null'] = test[['dss_row', 'dss_col']].sum(numeric_only=True, axis=1)  # css = dss1 + dss2. Alternatively we can use (dss1 + dss2)/2

    # calculating null values for synergies
    ## get unique drugs to be used in synergy null hypothesis construction
    r = test.drug_row_id.drop_duplicates().values
    c = test.drug_col_id.drop_duplicates().values
    u = np.unique(np.append(r, c))

    # calculate null hypothesis values for each drug in unique drugs
    null_hypothesis_values = {}
    for i in u:
        null_hypothesis_values[i] = calculate_synergy_null(i, test)

    test.rename(columns={'synergy_zip': 'synergy_zip_real', 'synergy_bliss': 'synergy_bliss_real',
                         'synergy_loewe': 'synergy_loewe_real', 'synergy_hsa': 'synergy_hsa_real'}, inplace=True)

    # use drug_row_id and drug_col_id to create a new 8 columns which will be used for null_hypothesis construction
    '''
    For predicting synergy, the additive model is defined differently, 
    the assumption is that the synergy of a drug combination is the sum of the single 'drug synergy'
    however, single drug synergy does not make sense, 
    that is why we need to make a new definition as the average synergy score of drug combinations that involve drug 1.
    '''
    t = test['drug_row_id'].map(null_hypothesis_values).apply(pd.Series)  # we are matching values from drug_row_id column using previously created dictionary
    w = test['drug_col_id'].map(null_hypothesis_values).apply(pd.Series)
    #  test['Null_synergy_row'] = test['Null_synergy_row'].apply(pd.Series)  # https://stackoverflow.com/questions/38231591/splitting-dictionary-list-inside-a-pandas-column-into-separate-columns
    test = pd.concat([test, t], axis=1)
    test.rename(columns={'synergy_zip': 'row_zip_null', 'synergy_bliss': 'row_bliss_null',
                         'synergy_loewe': 'row_loewe_null', 'synergy_hsa': 'row_hsa_null'}, inplace=True)

    test = pd.concat([test, w], axis=1)
    test.rename(columns={'synergy_zip': 'col_zip_null', 'synergy_bliss': 'col_bliss_null',
                         'synergy_loewe': 'col_loewe_null', 'synergy_hsa': 'col_hsa_null'}, inplace=True)

    # averaging synergies of row and col drugs. Final step in preping null hypothesis data
    test['zip_null'] = test[['row_zip_null', 'col_zip_null']].mean(numeric_only=True, axis=1)
    test['bliss_null'] = test[['row_bliss_null', 'col_bliss_null']].mean(numeric_only=True, axis=1)
    test['loewe_null'] = test[['row_loewe_null', 'col_loewe_null']].mean(numeric_only=True, axis=1)
    test['hsa_null'] = test[['row_hsa_null', 'col_hsa_null']].mean(numeric_only=True, axis=1)

    test.drop(to_drop, axis=1, inplace=True)
    test.set_index('block_id', inplace=True)

    # preparing for ML
    features = test.iloc[:, :2048].copy(deep=True)
    ZIP_real = test.loc[:, ['synergy_zip_real']].copy(deep=True)
    ZIP_null = test.loc[:, ['zip_null']].copy(deep=True)
    Bliss_real = test.loc[:, ['synergy_bliss_real']].copy(deep=True)
    Bliss_null = test.loc[:, ['bliss_null']].copy(deep=True)
    Loewe_real = test.loc[:, ['synergy_loewe_real']].copy(deep=True)
    Loewe_null = test.loc[:, ['loewe_null']].copy(deep=True)
    HSA_real = test.loc[:, ['synergy_hsa_real']].copy(deep=True)
    HSA_null = test.loc[:, ['hsa_null']].copy(deep=True)
    S_real = test.loc[:, ['S_score_real']].copy(deep=True)
    S_null = test.loc[:, ['S_score_null']].copy(deep=True)

    # holders of LR and naive RMSE's
    e1_ZIP = []
    e2_ZIP = []
    e1_Loewe = []
    e2_Loewe = []
    e1_HSA = []
    e2_HSA = []
    e1_Bliss = []
    e2_Bliss = []

    kfold_outer = KFold(n_splits=5, shuffle=True, random_state=2)
    counter = 0

    # holder indices
    indices = {}

    for train_index, test_index in kfold_outer.split(features):
        lr_split = LinearRegression(fit_intercept=False, n_jobs=7)

        # ZIP
        e1, e2 = compareLR(model=lr_split, train_index=train_index, test_index=test_index, features=features, target=ZIP_real, target_null=ZIP_null, lol=True)
        e1_ZIP.append(e1)  # LR
        e2_ZIP.append(e2)  # naive
#           print('ZIP lr RMSE:{0}, ZIP naive RMSE:{1}'.format(e1, e2))

        # Loewe
        e1, e2 = compareLR(model=lr_split, train_index=train_index, test_index=test_index, features=features, target=Loewe_real, target_null=Loewe_null,lol=True)
        e1_Loewe.append(e1)  # LR
        e2_Loewe.append(e2)  # naive
#           print('Loewe lr RMSE:{0}, Loewe naive RMSE:{1}'.format(e1, e2))

        # Bliss
        e1, e2 = compareLR(model=lr_split, train_index=train_index, test_index=test_index, features=features, target=Bliss_real, target_null=Bliss_null,lol=True)
        e1_Bliss.append(e1)  # LR
        e2_Bliss.append(e2)  # naive
#          print('Bliss lr RMSE:{0}, Bliss naive RMSE:{1}'.format(e1, e2))

        # HSA
        e1, e2 = compareLR(model=lr_split, train_index=train_index, test_index=test_index, features=features, target=HSA_real, target_null=HSA_null,lol=True)
        e1_HSA.append(e1)  # LR
        e2_HSA.append(e2)  # naive
#         print('HSA lr RMSE:{0}, HSA naive RMSE:{1}'.format(e1, e2))
        print('iteration: {0} on cell line: {1}'.format(counter+1, key))


        counter += 1
        indices[counter] = {'train': train_index, 'test': test_index}
    # cleaning = 0
    # for a,b,c,d in zip(e1_ZIP, e1_Loewe, e1_Bliss, e1_HSA):
    #
    #     if (t > 100 for t in [a, b, c, d]):
    #         print('iteration number {0}'.format(cleaning + 1), a, b, c, d)
    #         e1_ZIP.remove(a)
    #         e1_Loewe.remove(b)
    #         e1_Bliss.remove(c)
    #         e1_HSA.remove(d)
    #     cleaning += 1

    holder[value] = {'LR_ZIP': e1_ZIP, 'Naive_ZIP': e2_ZIP,
                     'LR_Loewe': e1_Loewe, 'Naive_Loewe': e2_Loewe,
                     'LR_Bliss': e1_Bliss, 'Naive_Bliss': e2_Bliss,
                     'LR_HSA': e1_HSA, 'Naive_HSA': e2_HSA,
                     'indices': indices}
#%%
# saving to pickle object
# with open('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/lr_rmse_27032019', 'wb') as f:
#     pickle.dump(holder, f, pickle.HIGHEST_PROTOCOL)
#
#%%

with open("/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/lr_rmse_27032019", "rb") as f:
    temp = pickle.load(f)  # done locally
tissues = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/cell_line.xlsx')
tissues = tissues[['name', 'tissue_name']]

#%% adds info about cell line and tissue to dict
for i, v in files_cellnames.items():

    print(i, v)

    print(tissues.loc[tissues['name'] == v, 'tissue_name'].values[0])
    temp[v]['Tissue'] = tissues.loc[tissues['name'] == v, 'tissue_name'].values[0]
    temp[v]['Cell_line'] = v
#%%
hold = pd.DataFrame(
    columns=['LR_ZIP', 'Naive_ZIP', 'LR_Loewe', 'Naive_Loewe', 'LR_Bliss', 'Naive_Bliss','LR_HSA','Naive_HSA', 'Tissue',
             'Cell_line'])
for i, v in temp.items():
    hold = hold.append(pd.DataFrame(v, columns=['LR_ZIP', 'Naive_ZIP', 'LR_Loewe', 'Naive_Loewe', 'LR_Bliss', 'Naive_Bliss','LR_HSA','Naive_HSA', 'Tissue',
             'Cell_line']))
hold.reset_index(drop=True, inplace=True)
print(hold.shape)
hold.drop(hold[hold.LR_ZIP > 100].index, inplace=True)
print(hold.shape)
#%%
hold = pd.melt(hold, id_vars=["Cell_line", 'Tissue'],
                value_vars=['LR_ZIP', 'Naive_ZIP', 'LR_Loewe', 'Naive_Loewe', 'LR_Bliss', 'Naive_Bliss','LR_HSA','Naive_HSA'])
hold = pd.get_dummies(hold, columns=['variable'])

#%% 'variable_LR_Bliss', 'variable_LR_HSA',
#       'variable_LR_Loewe', 'variable_LR_ZIP', 'variable_Naive_Bliss',
 #      'variable_Naive_HSA', 'variable_Naive_Loewe', 'variable_Naive_ZIP'
hold3 = hold.drop(['variable_LR_HSA',"variable_LR_Loewe",
                   'variable_LR_ZIP', 'variable_Naive_HSA', 'variable_Naive_Loewe', 'variable_Naive_ZIP'], axis = 1)  # BLISS
hold5 = hold.drop(['variable_LR_Bliss',"variable_LR_Loewe",
                   'variable_LR_ZIP', 'variable_Naive_Bliss', 'variable_Naive_Loewe', 'variable_Naive_ZIP'], axis=1)  # HSA
hold6 = hold.drop(['variable_LR_Bliss',"variable_LR_Loewe",
                   'variable_LR_HSA', 'variable_Naive_Bliss', 'variable_Naive_Loewe', 'variable_Naive_HSA'], axis=1)  # ZIP
hold9 = hold.drop(['variable_LR_Bliss',"variable_LR_ZIP",
                   'variable_LR_HSA', 'variable_Naive_Bliss', 'variable_Naive_ZIP', 'variable_Naive_HSA'], axis=1)  # Loewe

hold4 = hold3.drop(hold3[(hold3.variable_LR_Bliss == 0) & (hold3.variable_Naive_Bliss == 0)].index)
hold7 = hold5.drop(hold5[(hold5.variable_LR_HSA == 0) & (hold5.variable_Naive_HSA == 0)].index)
hold8 = hold6.drop(hold6[(hold6.variable_LR_ZIP == 0) & (hold6.variable_Naive_ZIP == 0)].index)

hold10 = hold9.drop(hold9[(hold9.variable_LR_Loewe == 0) & (hold9.variable_Naive_Loewe == 0)].index)


hold4 = hold4.drop(['variable_Naive_Bliss'], axis=1)
hold7 = hold7.drop(['variable_Naive_HSA'], axis=1)
hold8 = hold8.drop(['variable_Naive_ZIP'], axis=1)
hold10 = hold10.drop(['variable_Naive_Loewe'], axis=1)

#%%

fig, (ax1, ax2,ax3,ax4, ax5) = plt.subplots(5)

sns.violinplot(x="Tissue", y="value", ax=ax1, hue='variable_LR_Bliss', split=True, inner="quartile",
                    palette={0: "b", 1: "y"},
                    data=hold4, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
sns.despine(left=True)
sns.violinplot(x="Tissue", y="value", ax=ax2, hue='variable_LR_Loewe', split=True, inner="quartile",
                    palette={0: "b", 1: "y"},
                    data=hold10, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
sns.despine(left=True)
sns.violinplot(x="Tissue", y="value", ax=ax3, hue='variable_LR_HSA', split=True, inner="quartile",
                    palette={0: "b", 1: "y"},
                    data=hold7, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
sns.despine(left=True)
sns.violinplot(x="Tissue", y="value", ax=ax4, hue='variable_LR_ZIP', split=True, inner="quartile",
                    palette={0: "b", 1: "y"},
                    data=hold8, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
sns.despine(left=True)

# for the following plot, hold41 should be created in case3_viz.py
sns.violinplot(x="Tissue", y="value", ax=ax5, hue='variable_LR_test', split=True, inner="quartile",
                    palette={0: "b", 1: "y"},
                    data=hold41, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
sns.despine(left=True)

label_size = 88
ax1.set_ylabel("RMSE", fontsize=label_size)
ax2.set_ylabel("RMSE", fontsize=label_size)
ax3.set_ylabel("RMSE", fontsize=label_size)
ax4.set_ylabel("RMSE", fontsize=label_size)
ax5.set_ylabel("RMSE", fontsize=label_size)


ax5.set_xlabel('Category', fontsize=label_size)

tick_size = 76
ax1.tick_params(labelsize=tick_size)
ax1.grid(b=True, which='major', color='k', linestyle='-', alpha=0.15)
ax1.xaxis.grid(False)
ax1.set_ylim([0, 36])

ax2.tick_params(labelsize=tick_size)
ax2.grid(b=True, which='major', color='k', linestyle='-', alpha=0.15)
ax2.xaxis.grid(False)
ax2.set_ylim([0, 36])


ax3.tick_params(labelsize=tick_size)
ax3.grid(b=True, which='major', color='k', linestyle='-', alpha=0.15)
ax3.xaxis.grid(False)
ax3.set_ylim([0, 36])


ax4.tick_params(labelsize=tick_size)
ax4.grid(b=True, which='major', color='k', linestyle='-', alpha=0.15)
ax4.xaxis.grid(False)
ax4.set_ylim([0, 36])

ax5.tick_params(labelsize=tick_size)
ax5.grid(b=True, which='major', color='k', linestyle='-', alpha=0.15)
ax5.xaxis.grid(False)
ax5.set_ylim([0, 36])



ax1.xaxis.set_visible(False)
ax2.xaxis.set_visible(False)
ax3.xaxis.set_visible(False)
ax4.xaxis.set_visible(False)



blue_patch_HSA = mpatches.Patch(color='b', label='HSA additive model')
yellow_patch_HSA = mpatches.Patch(color='y', label='HSA linear regression')
blue_patch_ZIP = mpatches.Patch(color='b', label='ZIP additive model')
yellow_patch_ZIP = mpatches.Patch(color='y', label='ZIP linear regression')
blue_patch_Loewe = mpatches.Patch(color='b', label='Loewe additive model')
yellow_patch_Loewe = mpatches.Patch(color='y', label='Loewe linear regression')
blue_patch_Bliss = mpatches.Patch(color='b', label='Bliss additive model')
yellow_patch_Bliss = mpatches.Patch(color='y', label='Bliss linear regression')
blue_patch_CSS = mpatches.Patch(color='b', label='CSS additive model')
yellow_patch_CSS = mpatches.Patch(color='y', label='CSS inear regression')
#white_patch = mpatches.Patch(color=None, fc="w", fill=False, edgecolor='none', linewidth=0, label='92 cell lines')
#white2_patch = mpatches.Patch(color=None, fc="w", fill=False, edgecolor='none', linewidth=0, label='CV_in=5, CV_out=10')
#white1_patch = mpatches.Patch(color=None, fc="w", fill=False, edgecolor='none', linewidth=0, label='NaiveCSS=DSS1+DSS2')

x = 1
y = 1.2
# ax.legend(handles=[blue_patch, yellow_patch, white_patch, white2_patch, white1_patch], loc='upper right')
ax1.legend(handles=[blue_patch_Bliss, yellow_patch_Bliss], bbox_to_anchor=(x, y), fontsize=label_size)
ax2.legend(handles=[blue_patch_Loewe, yellow_patch_Loewe], bbox_to_anchor=(x, y), fontsize=label_size)
ax3.legend(handles=[blue_patch_HSA, yellow_patch_HSA], bbox_to_anchor=(x, y), fontsize=label_size)
ax4.legend(handles=[blue_patch_ZIP, yellow_patch_ZIP], bbox_to_anchor=(x, y), fontsize=label_size)
ax5.legend(handles=[blue_patch_CSS, yellow_patch_CSS], bbox_to_anchor=(x, y), fontsize=label_size)

fig.tight_layout()
#fig.subplots_adjust(top=0.88)

# fig.tight_layout()
# fig.subplots_adjust(top=0.88)
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = 'RMSE_4_synergies_' + now +'.pdf'




plt.show()
fig.savefig(filename, format='pdf')
plt.close(fig)
# to_save = plt.gcf()
# #to_save.savefig('sd of CSS values.png')
#%%
fig.savefig(filename, format='pdf')
plt.close(fig)

########################################
# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

counter_outer = 0
counter_inner = 0

# folds for inner CV
kfold = KFold(n_splits=3, shuffle=True, random_state=1)

# folds for outer CV
kfold_outer = KFold(n_splits=10, shuffle=True, random_state=2)

# fit on split train data

errors = {}
e1_list = []
e2_list = []

for train_index, test_index in kfold_outer.split(features):
    lr_split = LinearRegression(fit_intercept=False, n_jobs=7)
    lr_split.fit(features.iloc[train_index], ZIP_real.iloc[train_index])

    pred = lr_split.predict(features.iloc[test_index])
    e1 = np.sqrt(mean_squared_error(ZIP_real.iloc[test_index], pred))
    e2 = np.sqrt(mean_squared_error(ZIP_real.iloc[test_index], ZIP_null.iloc[test_index]))
    print('lr RMSE:{0}, naive RMSE:{1}'.format(e1,e2))
    e1_list.append(e1)
    e2_list.append(e2)
errors['e1'] = e1_list  # LR
errors['e2'] = e2_list  # naive

#%%




counter_outer += 1
    # dt is features, tt - target
    x_train, x_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = ZIP_real.iloc[train_index], ZIP_real.iloc[test_index]

    lr_split

    for train, validation in kfold.split(x_train):
        lr_split.fit(x_train.iloc[train], y_train.iloc[train])
        train_error, validation_error, test_error = calc_metrics_RMSE(x_train.iloc[train], y_train.iloc[train],
                                                                      x_train.iloc[validation], y_train.iloc[validation],
                                                                      x_test, y_test,
                                                                      lr_split)
        naive_test_error = np.sqrt(mean_squared_error(y_test, ZIP_null.iloc[test_index]))

        if train_error == -1:
            manual = [train_index, test_index, train, validation]
            break


        print('iteration # {0}, with train error = {1}, validation error = {2} and test error = {3}. \n naive model error on test fold is {4}'
              .format(counter_inner, train_error, validation_error, test_error, naive_test_error))
        errors[counter_inner] = [train_error, validation_error, test_error, naive_test_error]
        counter_inner += 1

#%%
'''
d.update((k, v * 0.5) for k,v in d.items())
https://stackoverflow.com/questions/15536623/modify-dict-values-inplace

{k: v for k, v in hand.items() if v != 0}
https://stackoverflow.com/questions/15158599/removing-entries-from-a-dictionary-based-on-values
'''

# this rounds to three (3) significant figures all the values in the dict, given that they are not negative (ie RMSE >> 1000)
# is there mb a better way to be doing that?

cp = {key: [round(v, 3) for v in item] for key, item in errors.items() if -1 not in item}
# errors.update((k,[round(item, 3) for item in v]) for k, v in errors.items() if -1 ] not)


#%%
'''
this is taken from here
https://stackoverflow.com/questions/43357274/separate-halves-of-split-violinplot-to-compare-tail-data
'''

def offset_violinplot_halves(ax, delta, width, inner, direction):
    """
    This function offsets the halves of a violinplot to compare tails
    or to plot something else in between them. This is specifically designed
    for violinplots by Seaborn that use the option `split=True`.

    For lines, this works on the assumption that Seaborn plots everything with
     integers as the center.

    Args:
     <ax>    The axis that contains the violinplots.
     <delta> The amount of space to put between the two halves of the violinplot
     <width> The total width of the violinplot, as passed to sns.violinplot()
     <inner> The type of inner in the seaborn
     <direction> Orientation of violinplot. 'hotizontal' or 'vertical'.

    Returns:
     - NA, modifies the <ax> directly
    """
    # offset stuff
    if inner == 'sticks':
        lines = ax.get_lines()
        for line in lines:
            if direction == 'horizontal':
                data = line.get_ydata()
                print(data)
                if int(data[0] + 1)/int(data[1] + 1) < 1:
                    # type is top, move neg, direction backwards for horizontal
                    data -= delta
                else:
                    # type is bottom, move pos, direction backward for hori
                    data += delta
                line.set_ydata(data)
            elif direction == 'vertical':
                data = line.get_xdata()
                print(data)
                if int(data[0] + 1)/int(data[1] + 1) < 1:
                    # type is left, move neg
                    data -= delta
                else:
                    # type is left, move pos
                    data += delta
                line.set_xdata(data)


    for ii, item in enumerate(ax.collections):
        # axis contains PolyCollections and PathCollections
        if isinstance(item, matplotlib.collections.PolyCollection):
            # get path
            path, = item.get_paths()
            vertices = path.vertices
            half_type = _wedge_dir(vertices, direction)
            # shift x-coordinates of path
            if half_type in ['top', 'bottom']:
               if inner in ["sticks", None]:
                    if half_type == 'top': # -> up
                        vertices[:, 1] -= delta
                    elif half_type == 'bottom': # -> down
                        vertices[:, 1] += delta
            elif half_type in ['left', 'right']:
                if inner in ["sticks", None]:
                    if half_type == 'left': # -> left
                        vertices[:,0] -= delta
                    elif half_type == 'right': # -> down
                        vertices[:,0] += delta

def _wedge_dir(vertices, direction):
    """
    Args:
      <vertices>  The vertices from matplotlib.collections.PolyCollection
      <direction> Direction must be 'horizontal' or 'vertical' according to how
                   your plot is laid out.
    Returns:
      - a string in ['top', 'bottom', 'left', 'right'] that determines where the
         half of the violinplot is relative to the center.
    """
    if direction == 'horizontal':
        result = (direction, len(set(vertices[1:5,1])) == 1)
    elif direction == 'vertical':
        result = (direction, len(set(vertices[-3:-1,0])) == 1)
    outcome_key = {('horizontal', True): 'bottom',
                   ('horizontal', False): 'top',
                   ('vertical', True): 'left',
                   ('vertical', False): 'right'}
    # if the first couple x/y values after the start are the same, it
    #  is the input direction. If not, it is the opposite
    return outcome_key[result]

# create some data
n = 100 # number of samples
c = ['cats', 'rats', 'bears', 'pears', 'snares'] # classes
y = np.random.randn(n)
x = np.random.choice(c, size=n)
z = np.random.rand(n) > 0.5 # sub-class
data = pd.DataFrame(dict(x=x, y=y, z=z))
print('done making data')

# initialise new axes;
fig, (ax1, ax2) = plt.subplots(2)

inner = "sticks" # Note: 'box' is default
width = 0.75
delta = 0.05
final_width = width - delta
print(data)
sns.violinplot(data=data, x='y', y='x',
               split=True, hue = 'z',
               ax = ax1, inner='sticks',
               bw = 0.2)
sns.violinplot(data=data, x='x', y='y',
               split=True, hue = 'z',
               ax = ax2, inner='sticks',
               bw = 0.2)

offset_violinplot_halves(ax1, delta, final_width, inner, 'horizontal')
offset_violinplot_halves(ax2, delta, final_width, inner, 'vertical')

plt.show()

#%%

fig, (ax1, ax2) = plt.subplots(2)

inner = "sticks" # Note: 'box' is default
width = 0.75
delta = 0.05
final_width = width - delta

sns.violinplot(x="Tissue", y="value", ax=ax1, hue='variable_LR_Bliss', split=True,
                    palette={0: "b", 1: "y"},
                    data=hold4, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
sns.violinplot(x="Tissue", y="value", ax=ax2, hue='variable_LR_Bliss', split=True,
                    palette={0: "b", 1: "y"},
                    data=hold4, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
offset_violinplot_halves(ax1, delta, final_width, inner, 'horizontal')
offset_violinplot_halves(ax2, delta, final_width, inner, 'vertical')
plt.show()