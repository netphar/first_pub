# %%
# for some reason this files results in PyCharm running out of memory. Made me increase memory allocation limit to 1.5 Gb. I guess it is related to IDE itself
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
#%%
# Loads data from trained LinR models.
with open("/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/lr_r2_all_noT98G.pickle", "rb") as f:
    lr_r2_all_noT98G = pickle.load(f)  # done locally
with open("/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/lr_rmse_all_noT98G.pickle", "rb") as f:
    lr_rmse_all_noT98G = pickle.load(f)  # done locally
with open("/Users/zagidull/Desktop/lr_r2_all.pickle", "rb") as f:
    lr_r2_all = pickle.load(f)  # done on atlas
with open("/Users/zagidull/Desktop/lr_rmse_all.pickle", "rb") as f:
    lr_rmse_all = pickle.load(f)  # done on atlas

#%%
# get list of SMILES files saved in csv. Add tissue information
directory = os.fsencode('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/averages_17012019/')

to_drop = ['drug_row', 'drug_col', 'block_id', 'drug_row_id', 'drug_col_id', 'cell_line_id', 'synergy_zip',
           'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'ic50_row', 'ic50_col', 'cell_line_name', 'smiles_row',
           'smiles_col']

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

urgh_cleaned = {}
for file in urgh:
    s = file
    match = re.findall("_(.*?)_", s)
    urgh_cleaned[s] = match[0]
#    print(match[0])
print(len(urgh_cleaned))

tissues = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/cell_line.xlsx')
tissues = tissues[['name', 'tissue_name']]

lr_rmse_all_noT98GTissueCell = lr_rmse_all_noT98G.copy()
for i, v in urgh_cleaned.items():
    print(i, v)
    if v == 'NCI':
        v = 'NCI/ADR-RES'
    print(tissues.loc[tissues['name'] == v, 'tissue_name'].values[0])
    lr_rmse_all_noT98GTissueCell[i]['Tissue'] = tissues.loc[tissues['name'] == v, 'tissue_name'].values[0]
    lr_rmse_all_noT98GTissueCell[i]['Cell_line'] = v

#%%
# get training results
hold = pd.DataFrame(
    columns=['LR_training', 'Naive_training', 'LR_validation', 'Naive_validation', 'LR_test', 'Naive_test', 'Tissue',
             'Cell_line'])
for i, v in lr_rmse_all_noT98GTissueCell.items():
    hold = hold.append(pd.DataFrame(v, columns=['LR_training', 'Naive_training', 'LR_validation', 'Naive_validation',
                                                'LR_test', 'Naive_test', 'Tissue', 'Cell_line']))

hold.reset_index(drop=True, inplace=True)
print(hold.shape)
hold.drop(hold[hold.LR_validation > 100].index, inplace=True)
print(hold.shape)
#hold.drop(hold[hold.LR_validation < -1].index, inplace=True)
#print(hold.shape)

hold = pd.melt(hold, id_vars=["Cell_line", 'Tissue'],
                value_vars=['LR_training', 'Naive_training', 'LR_validation',
                            'Naive_validation', 'LR_test', 'Naive_test'])
hold = pd.get_dummies(hold, columns=['variable'])

#%%
# data cleanup
hold31 = hold.drop(['variable_LR_training',"variable_LR_validation",
                   'variable_Naive_training', 'variable_Naive_validation'], axis = 1)
hold51 = hold.drop(['variable_LR_training',"variable_LR_test",
                    'variable_Naive_training', 'variable_Naive_test'], axis=1)
hold61 = hold.drop(['variable_LR_test',"variable_LR_validation",
                    'variable_Naive_test', 'variable_Naive_validation'], axis=1)

hold41 = hold31.drop(hold31[(hold31.variable_LR_test == 0) & (hold31.variable_Naive_test == 0)].index)
hold71 = hold51.drop(hold51[(hold51.variable_LR_validation == 0) & (hold51.variable_Naive_validation == 0)].index)
hold81 = hold61.drop(hold61[(hold61.variable_LR_training == 0) & (hold61.variable_Naive_training == 0)].index)

hold41 = hold41.drop(['variable_Naive_test'], axis=1)
hold71 = hold71.drop(['variable_Naive_validation'], axis=1)
hold81 = hold81.drop(['variable_Naive_training'], axis=1)

#%%
# here we get a list of csv files with bitwise averages
# duplicate of block from linenum = 23 onwards

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
    it populates
    :param x: input df with all cell lines
    :param name: cell line name
    :return: filters list of drugs without ARSENIC TRIOXIDE, since for these SMILES generated are not considered to be valid
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


def compareLR(model, train_index, test_index, features, target, target_null):
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

    return e1, e2


#%%
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
    that is why we need to make a new definition as the average synergy score of all drug combinations that involve drug 1.
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
# saving to pickle object holder
# it contains 'LR_ZIP', 'Naive_ZIP', 'LR_Loewe', 'Naive_Loewe', 'LR_Bliss', 'Naive_Bliss', 'LR_HSA', 'Naive_HSA', 'indices' from 5-fold CV using Linear Regression
# with open('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/lr_rmse_27032019', 'wb') as f:
#     pickle.dump(holder, f, pickle.HIGHEST_PROTOCOL)
#
#%%
# this basically loads the file created in the massive loops of training and reading in the csv files

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

# this part here deletes outliers with RMSE above 100
hold.drop(hold[hold.LR_ZIP > 100].index, inplace=True)
print(hold.shape)
#%%
# prep for violinplot drawing
hold = pd.melt(hold, id_vars=["Cell_line", 'Tissue'],
                value_vars=['LR_ZIP', 'Naive_ZIP', 'LR_Loewe', 'Naive_Loewe', 'LR_Bliss', 'Naive_Bliss','LR_HSA','Naive_HSA'])
hold = pd.get_dummies(hold, columns=['variable'])

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
# violin plot drawing
# ends up saving file 6000x6000 px with 5 violin plots of RMSE values

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
#fig.savefig(filename, format='pdf')
# plt.close(fig)