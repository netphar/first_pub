#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import category_scatter
from collections import defaultdict
import seaborn as sns

#%%
'''
NB: if a drug is administered as a monotherapy in the screen, nan is given in the second column
'''
dtypes = {'SCREENER': 'object', 'STUDY': 'object', 'TESTDATE': 'str', 'PLATE': 'object', 'NSC1': 'Int64', 'NSC2': 'Int64'}
# parse_dates = ['TESTDATE']
file = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017.csv', sep=',',
                   dtype=dtypes) #, parse_dates=parse_dates)
file1 = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_imputed.csv', sep=',',
                   dtype=dtypes) #, parse_dates=parse_dates)
original = file.copy()  # this is a copy of the file just in case

#%%
file.fillna(pd.Series(-9999, index=file.select_dtypes(exclude='category').columns), inplace=True)
file_grouped = file.groupby(['NSC1', 'NSC2', 'CELLNAME'])

# also need to get na's removed from file1
file1.fillna(pd.Series(-9999, index=file1.select_dtypes(exclude='category').columns), inplace=True)


#%% we are checking if any of the drug1 - drug2 - cell line are tested in more than one plate
single_plate_per_combo = 0
single_plate_per_combo_holder = []
many_plate_per_combo = 0
many_plate_per_combo_holder = []

for i,v in file_grouped:
    if i[1] != -9999:
        if len(v.PLATE.unique()) == 1:
            single_plate_per_combo += 1
            single_plate_per_combo_holder.append(i)

        else:
            many_plate_per_combo += 1
            many_plate_per_combo_holder.append(i)

#%% let's get # of plates for combos tested in multiple plates
sets = set()
for i in many_plate_per_combo_holder:
    sets.add(file_grouped.get_group(i).PLATE.unique().shape[0])

#%% we are checking how many good plates are there vs bad plates
# good plate is where drug1-drug2-cell line is tested in the same plate with a single drug
file_grouped_plate = file.groupby(['PLATE'])

good_plate = 0
good_plate_holder = []
bad_plate = 0
bad_plate_holder = []

for i,v in file_grouped_plate:
    gr = v.groupby(['NSC1', 'NSC2', 'CELLNAME'])
    k = gr.groups.keys()
    for o,_ in gr:
        if (o[0], -9999, o[2]) in k:
            good_plate_holder.append(i)
            good_plate += 1
        else:
            bad_plate += 1
            bad_plate_holder.append(i)

#%% let's calculate sd of PERCENTGROWTH for single drugs per cell line per dose

grouped = file.groupby(['NSC1', 'NSC2'])
d = defaultdict(list)  # using defaultdict allows to use .append method to add values to a key

for i,v in grouped:
    if i[1] == -9999:
        temp = v.groupby(['CELLNAME', 'CONC1'])


#        temp = v.groupby(['CONC1'])
        temp1 = []
        for _,k in temp:
            temp1.append(k['PERCENTGROWTH'].std(ddof=1))
        d[i[0]].append(temp1)


#%% let's plot the stuff above. So drug on x axis and sd on y axis
dd = pd.DataFrame.from_dict(d, orient='index', columns=['sd'])
dd.reset_index(level=0, inplace=True)
dd.rename(index=str, columns={"index": "drug"}, inplace=True)
dd = dd['sd'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('sd').join(dd[['drug']], how='left')

sns.catplot(x='drug', y='sd', kind='bar', data=dd, ci='sd')
fig = plt.gcf()
fig.set_size_inches(20,15)
plt.xticks(rotation='vertical')
plt.title('sd of single drug screens')
plt.show()


#%% let's calculate sd of PERCENTGROWTHNOTZ for single drugs per cell line per dose.
# change d and grouped to d1 and grouped1 and so on
# this is with PERCENTGROWTHNOTZ imputed using r-script sharing_notebooks/cleanup/NCI_processing.R

grouped1 = file1.groupby(['NSC1', 'NSC2'])
d1 = defaultdict(list)  # using defaultdict allows to use .append method to add values to a key

for i,v in grouped1:
    if i[1] == -9999:
        temp = v.groupby(['CELLNAME', 'CONC1'])


#        temp = v.groupby(['CONC1'])
        temp1 = []
        for _,k in temp:
            temp1.append(k['PERCENTGROWTHNOTZ'].std(ddof=1))
        d1[i[0]].append(temp1)


dd1 = pd.DataFrame.from_dict(d1, orient='index', columns=['sd'])
dd1.reset_index(level=0, inplace=True)
dd1.rename(index=str, columns={"index": "drug"}, inplace=True)
dd1 = dd1['sd'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('sd').join(dd1[['drug']], how='left')

sns.catplot(x='drug', y='sd', kind='bar', data=dd1, ci='sd')
fig = plt.gcf()
fig.set_size_inches(20,15)
plt.xticks(rotation='vertical')
plt.title('sd of single drug screens. with PERCENTGROWTHNOTZ imputed')
plt.show()

#%% to test how many examples are there of plates where monotherapy is not on the same plate as the combination
counter = 0
for i,v in file_grouped:
    if i[1] != -9999:
        q = file_grouped.get_group((i[0], -9999, i[2]))
        w = file_grouped.get_group((i[1], -9999, i[2]))
        print(i, q, w)
        counter += 1
        if counter == 1:
            break


#%%
test = file.loc[((file.NSC1 == 256439) & (file.NSC2 == 740) & (file.CELLNAME == 'HCT-15')) |
                ((file.NSC1 == 740) & (file.NSC2 == 256439) & (file.CELLNAME == 'HCT-15')) |
                ((file.NSC1 == 740) & (file.NSC2 == -9999) & (file.CELLNAME == 'HCT-15')) |
                ((file.NSC1 == 256439) & (file.NSC2 == -9999) & (file.CELLNAME == 'HCT-15'))]

#%%
test_grouped = test.groupby(['NSC1', 'NSC2'])

a = test_grouped.get_group((256439, 740))
b = test_grouped.get_group((256439, -9999))
c = test_grouped.get_group((740, -9999))

category_scatter(x='PLATE', y='PERCENTGROWTH', label_col='CONC1', data=b)
fig = plt.gcf()
fig.set_size_inches(15,10)
plt.xticks(rotation='vertical')
plt.title((256439, -9999))
plt.show()

category_scatter(x='PLATE', y='PERCENTGROWTH', label_col='CONC1', data=c)
fig = plt.gcf()
fig.set_size_inches(15,10)
plt.xticks(rotation='vertical')
plt.title((740, -9999))
plt.show()

d = defaultdict(list)
for i in c.CONC1.unique():
    d[i].append(c.loc[c.CONC1 == i].PERCENTGROWTH.values)

    c.loc[c.CONC1 == i][['PLATE', 'PERCENTGROWTH']]

c.groupby(['CONC1'])['PERCENTGROWTH'].aggregate(lambda x: np.std(x, ddof=0))


#%%

for i,v in test_grouped:
    print(i)
    v.PERCENTGROWTH
#%%
holder = []
counter = 0
for i,v in file_grouped:
    # holder.append(v['PLATE'].unique().values)
    # counter += 1
    if (len(v['PLATE'].unique()) > 1) & (

    ):
        print(v['PLATE'].unique())
        counter += 1