#%% initial import
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle

from func import locing

sns.set(style="seaborn-colorblind")
sns.set(rc={'figure.figsize': (15.7, 7.27)})

#with open('filter_input.pickle', 'wb') as f:
#    pickle.dump(key_holder, f, pickle.HIGHEST_PROTOCOL)

with open('/Users/zagidull/PycharmProjects/test_scientific/filter_input.pickle', 'rb') as f:
    data = pickle.load(f)
#%% unpacking the dict to separate into single or multiple studies
key_holder = data
counter = 0
single_sd = []
multiple_sd = []
al = []
on = []
keys = []
for key, value in key_holder.items():
    if value['type'] == 'single_study':
        single_sd.append(value['sd'])
    else:
        keys.append(key)
        multiple_sd.append(value['sd_cell_line'])
        al.append(value['study_wide']['ALMANAC']['sd_study_wide'])
        on.append(value['study_wide']['ONEIL']['sd_study_wide'])

# we have to use list(set()), since there is a lot of repeated elements, as we are searching several times for the same drug combo.
# It happens because we are removing the cell_line part, and using study instead. Which results in that behavoir
al = list(set(al))
on = list(set(on))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
sns.distplot(single_sd, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='blue',
             label='Single' + ' number of datapoints: ' + str(len(single_sd)))
sns.distplot(multiple_sd, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='red',
             label='multiple' + ' number of datapoints: ' + str(len(multiple_sd)))
sns.distplot(al, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='green',
             label='ALMANAC' + ' number of datapoints: ' + str(len(al)))
sns.distplot(on, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='yellow',
             label='ONEIL' + ' number of datapoints: ' + str(len(on)))

plt.title('sd of CSS values')
plt.ylabel('Density')
plt.xlabel('sd')
to_save = plt.gcf()
plt.show()
to_save.savefig('sd of CSS values.png')
#%%  just picking up examples that are present in multiple studies and sorting them from highest to lowest
for_sorting = {}
for key, value in key_holder.items():
    if value['type'] == 'multiple_study':
        for_sorting[key] = value['sd_cell_line']
out = sorted(for_sorting.items(), key=lambda kv: kv[1], reverse=True)
out = out[:150]  # taking 25% of key:value pairs with the highets sd values
#%%
#a = pd.DataFrame()
drugs = []
cell_lines = []
studies = []
for i in out:
    a = locing(input_all, i[0][0], i[0][1], i[0][2])
    row = a['drug_row'].drop_duplicates().to_string(index=False)
    col = a['drug_col'].drop_duplicates().to_string(index=False)
    cell = a['cell_line_name'].drop_duplicates().to_string(index=False)
    study = set(a['study'])
    studies.append(study)
    drugs.append(row)
    drugs.append(col)
    cell_lines.append(cell)
    print(row, col, cell, study)
studies1 = []
for i in studies:
    i = list(i)
    studies1.append(i)
studies1 = [x for sublist in studies1 for x in sublist]
#%%
g = sns.countplot(drugs)
plt.xticks(rotation=90)
plt.show()
g = sns.countplot(cell_lines)
plt.xticks(rotation=90)
plt.show()
g = sns.countplot(studies1)
plt.xticks(rotation=90)
plt.show()


#%%
#single_sd = []
#multiple_sd = []
#al = {}
#on = {}
#keys = []
#for key, value in key_holder.items():
#    if value['type'] == 'single_study':
#        single_sd.append(value['sd'])
#    else:
#        keys.append(key)
#        multiple_sd.append(value['sd_cell_line'])
#       al[key] = value['study_wide']['ALMANAC']['sd_study_wide']
#       on[key] = value['study_wide']['ONEIL']['sd_study_wide']


#%% drawing pretty pictures
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# sns.distplot(al, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
#              hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1},
#              bins='auto', color='blue', label='ALMANAC' + ' number of datapoints: ' + str(len(al)))
# sns.distplot(on, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
#              hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1},
#              bins='auto', color='darkred', label='ONEIL' + ' number of datapoints: ' + str(len(on)))
#
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# textstr = '\n'.join((
#     'ALMANAC = multiples only from NCI across all cell_lines',
#     'ONEIL = multiples only from O\'Neil across all cell_lines'
# ))
#
# plt.title('Standard deviation of CSS values.')
# # ax.text(44.5, .125,'bipositional = reversing drug row and drug col finds occurences', fontsize=9) #add text
# # ax.text(44.5, .132,'unipositional = reversing drug row and drug col finds NO occurences', fontsize=9) #add text
# # ax.text(44.5, .146,'singles = present in one study', fontsize=9) #add text
# ax.text(.712, .75, textstr, fontsize=10, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
#         bbox=props)  # add text
#
# plt.ylabel('Density')
# plt.xlabel('std')
# plt.show()

#%%
