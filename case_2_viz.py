#%% initial import
import numpy as np
from scipy.stats import ttest_ind
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import datetime
from func import locing

# sns.set(style="ticks")
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
#%%
sns.set_style("whitegrid")
sns.set_context('paper')

al = list(set(al))
on = list(set(on))
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

sns.distplot(single_sd, ax=ax1, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='blue',
             label='# combinations found in single study: ' + str(len(single_sd)))
sns.distplot(multiple_sd, ax=ax1, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='red',
             label='# combinations present in multiple studies: ' + str(len(multiple_sd)))
sns.distplot(al, ax=ax2, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='green',
             label='# combinations found in ALMANAC: ' + str(len(al)))
sns.distplot(on, ax=ax2, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='violet',
             label='# combinations found in O\'NEIL: ' + str(len(on)))

ax1.legend(fontsize=10, loc='upper right')
ax2.legend(fontsize=10, loc='upper right')

ax1.tick_params(labelsize=10)
ax2.tick_params(labelsize=10)

ax1.grid(b=True, which='major', color='k', linestyle='--', alpha=0.1)
ax2.grid(b=True, which='major', color='k', linestyle='--', alpha=0.1)



fig.suptitle('SD of CSS values of drugA & drugB combinations', fontsize=18)
ax1.set_title('cell line specific', fontsize=14)
ax2.set_title('found in multiple studies / study specific / not cell line specific', fontsize=14)

ax1.set_ylabel('Density', fontsize=12)
ax1.set_xlabel('SD', fontsize=12)
ax2.set_xlabel('SD', fontsize=12)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = 'case2_' + now +'.png'

#plt.show()
# to_save = plt.gcf()
# #to_save.savefig('sd of CSS values.png')
fig.savefig(filename, dpi=600)
plt.close(fig)


#%% for jing's presentation at SLAS
sns.set_style("whitegrid")
sns.set_context('paper')

al = list(set(al))
on = list(set(on))
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)

fig, ax = plt.subplots(ncols=1)

sns.distplot(single_sd, ax=ax, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='blue',
             label='in single study: ' + str(len(single_sd)))
sns.distplot(multiple_sd, ax=ax, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='red',
             label='in multiple studies: ' + str(len(multiple_sd)))
sns.distplot(al, ax=ax, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='green',
             label='in ALMANAC, all cell lines: ' + str(len(al)))
sns.distplot(on, ax=ax, hist=False, kde=True, rug=False, kde_kws={'linewidth': 2},
             hist_kws={"histtype": "step", 'linewidth': 1, "alpha": 1}, bins='auto', color='violet',
             label='in O\'NEIL, all cell lines: ' + str(len(on)))

ax.legend(fontsize=20, loc='upper right')

ax.tick_params(labelsize=20)

ax.grid(b=True, which='major', color='k', linestyle='--', alpha=0.1)



fig.suptitle('SD of CSS values of drugA & drugB & cell line combinations', fontsize=30)
#ax1.set_title('cell line specific', fontsize=14)
#ax2.set_title('found in multiple studies / study specific / not cell line specific', fontsize=14)

ax.set_ylabel('Density', fontsize=24)
ax.set_xlabel('SD', fontsize=24)
#ax2.set_xlabel('SD', fontsize=12)

fig.tight_layout()
fig.subplots_adjust(top=0.88)
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = 'case2_merged_' + now +'.pdf'

#plt.show()
# to_save = plt.gcf()
# #to_save.savefig('sd of CSS values.png')
fig.savefig(filename, dpi=600)
plt.close(fig)



# #%%  just picking up examples that are present in multiple studies and sorting them from highest to lowest
# for_sorting = {}
# for key, value in key_holder.items():
#     if value['type'] == 'multiple_study':
#         for_sorting[key] = value['sd_cell_line']
# out = sorted(for_sorting.items(), key=lambda kv: kv[1], reverse=True)
# out = out[:150]  # taking 25% of key:value pairs with the highets sd values
# #%%
# #a = pd.DataFrame()
# drugs = []
# cell_lines = []
# studies = []
# for i in out:
#     a = locing(input_all, i[0][0], i[0][1], i[0][2])
#     row = a['drug_row'].drop_duplicates().to_string(index=False)
#     col = a['drug_col'].drop_duplicates().to_string(index=False)
#     cell = a['cell_line_name'].drop_duplicates().to_string(index=False)
#     study = set(a['study'])
#     studies.append(study)
#     drugs.append(row)
#     drugs.append(col)
#     cell_lines.append(cell)
#     print(row, col, cell, study)
# studies1 = []
# for i in studies:
#     i = list(i)
#     studies1.append(i)
# studies1 = [x for sublist in studies1 for x in sublist]
#%%
# g = sns.countplot(drugs)
# plt.xticks(rotation=90)
# plt.show()
# g = sns.countplot(cell_lines)
# plt.xticks(rotation=90)
# plt.show()
# g = sns.countplot(studies1)
# plt.xticks(rotation=90)
# plt.show()


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
