import datetime
import os
import pickle
import re

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sns.set(rc={'figure.figsize': (15.7, 7.27)}, color_codes=True)
sns.set_style("whitegrid")
sns.set_context('paper')

#%%
# scp -P 22 bzagidul@atlas.fimm.fi:/homes/bzagidul/test_grun/linR/averages_17012019/l* ~/Desktop/

with open("/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/lr_r2_all_noT98G.pickle", "rb") as f:
    lr_r2_all_noT98G = pickle.load(f)  # done locally
with open("/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/lr_rmse_all_noT98G.pickle", "rb") as f:
    lr_rmse_all_noT98G = pickle.load(f)  # done locally
with open("/Users/zagidull/Desktop/lr_r2_all.pickle", "rb") as f:
    lr_r2_all = pickle.load(f)  # done on atlas
with open("/Users/zagidull/Desktop/lr_rmse_all.pickle", "rb") as f:
    lr_rmse_all = pickle.load(f)  # done on atlas

#%%
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
# ax = sns.violinplot(x="Tissue", y="value", hue='variable_LR_test', split=True, inner="quartile",palette={0: "b", 1: "y"},
#                orient='h',data=hold)
# sns.despine(left=True)
# ax.set_title('RMSE of predicted CSS vs actual CSS. LinR vs Naive on validation fold. All cell lines (except T98G). CV_out=5. CV_in=10. NaiveCSS = DSS1 + DSS2')
#
# plt.ylabel("RMSE")
# plt.xlabel('category')
# plt.show()

#%%
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

fig, ax = plt.subplots(ncols=1)

sns.violinplot(x="Tissue", y="value", ax=ax, hue='variable_LR_test', split=True, inner="quartile",
                    palette={0: "b", 1: "y"},
                    data=hold41, scale='count',
               order=["brain", "breast", "colon", "hem_lymph", "kidney", "large_intest", "lung", "ovary", "prostate", "skin"])
sns.despine(left=True)


ax.set_ylabel("RMSE", fontsize=24)
ax.set_xlabel('Category', fontsize=24)
ax.tick_params(labelsize=20)
ax.grid(b=True, which='major', color='k', linestyle='-', alpha=0.15)
ax.xaxis.grid(False)



blue_patch = mpatches.Patch(color='b', label='Additive model')
yellow_patch = mpatches.Patch(color='y', label='Linear Regression')
#white_patch = mpatches.Patch(color=None, fc="w", fill=False, edgecolor='none', linewidth=0, label='92 cell lines')
#white2_patch = mpatches.Patch(color=None, fc="w", fill=False, edgecolor='none', linewidth=0, label='CV_in=5, CV_out=10')
#white1_patch = mpatches.Patch(color=None, fc="w", fill=False, edgecolor='none', linewidth=0, label='NaiveCSS=DSS1+DSS2')

# ax.legend(handles=[blue_patch, yellow_patch, white_patch, white2_patch, white1_patch], loc='upper right')
ax.legend(handles=[blue_patch, yellow_patch], bbox_to_anchor=(1, 1.1), fontsize=20)


fig.tight_layout()
fig.subplots_adjust(top=0.88)
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = 'case3_' + now +'.pdf'

plt.show()
# to_save = plt.gcf()
# #to_save.savefig('sd of CSS values.png')
#%%
fig.savefig(filename, format='pdf')
plt.close(fig)

# plt.show()
#%%
#
# sns.kdeplot(lr_r2_all_noT98G['_786-0_smiles_17012019.csv']['LR_test'], label='LR_test_old')
# sns.kdeplot(lr_r2_all['_786-0_smiles_17012019.csv']['LR_test'], label='LR_test_new')
# #sns.kdeplot(lr_r2_all_noT98G['_786-0_smiles_17012019.csv']['Naive_test'], label='Naive_test_old')
# #sns.kdeplot(lr_r2_all['_786-0_smiles_17012019.csv']['Naive_test'], label='Naive_test_new')
# plt.legend()
# plt.show()
