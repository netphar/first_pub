#%%
import pandas as pd
import numpy as np
#%%
'''
NB: if a drug is administered as a monotherapy in the screen, nan is given in the second column
'''
dtypes = {'SCREENER': 'category', 'STUDY': 'category', 'TESTDATE': 'str', 'PLATE': 'category', 'NSC1': 'Int64', 'NSC2': 'Int64'}
# parse_dates = ['TESTDATE']
file = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017.csv', sep=',',
                   dtype=dtypes) #, parse_dates=parse_dates)
#%% pick only those that contain NA
noNA = file.loc[~file.isnull().any(axis=1)]
withNA = file.loc[file.isnull().any(axis=1)]
#%% testing if any of the combos are tested on separate plates / test centers

noNA_grouped = noNA.groupby(['NSC1','NSC2', 'CELLNAME'])

holder = []
count = 0
for i,v in noNA_grouped:  # get number of unique combos where
    if (len(v['PLATE'].unique()) > 1) | (len(v['SCREENER'].unique()) > 1) | (len(v['STUDY'].unique()) > 1):
        count += 1
        holder.append(i)
#%% this step is necessary, as pandas ignores rows with na's
# file.fillna(-999, inplace=True)  # this fails, since there are categorical columns present. To make it work as is, it is necessary to add the missing value to categories
withNA.fillna(pd.Series(-9999, index=file.select_dtypes(exclude='category').columns), inplace=True)

withNA_grouped = withNA.groupby(['NSC1','NSC2','CELLNAME'])
#%%
holder1 = []
count1 = 0
for i,v in withNA_grouped:  # get number of unique combos where
    if (len(v['PLATE'].unique()) > 1) | (len(v['SCREENER'].unique()) > 1) | (len(v['STUDY'].unique()) > 1):
        count1 += 1
        holder1.append(i)
#%%
for i,v in grouped:
    if len(v['PLATE'].unique()) >1:
        print(i,v)
        break
#%%

plates = file.PLATE.unique()
grouped = file.groupby(['PLATE'])  # groupby by default ignores rows with na's.

# so let's try to fill in na's / nan's using fillna
for i,v in grouped:
    if v.isnull().any().any():
        print(i,v)
        # in order to fillna in all the columns to work, value used to fill na's needs to be present in categories if any of the data is categorical
        # next line selects all rows that are categorical and applies a lambda function adding missing value as category
        v.loc[:,list(v.select_dtypes(include='category').columns)] = v.select_dtypes(include='category').apply(lambda x: x.cat.add_categories([-999]), axis=0)
        v.fillna(-999, inplace=True)
        break
#%%

test = file.loc[file.PLATE == 'A498_1_T72'].groupby(['NSC1','NSC2','CELLNAME'])

#%%
count = 0
for i,v in grouped:
    if len(v['NSC1'].unique()) + len(v['NSC2'].unique()) + len(v['CELLNAME'].unique()) > 3:
        print(i,v)
        count += 1
        if count >2:
            break
#%%
file_noNA = file.dropna(axis=0)  # dropping all na rows
file_noSingleDrug = file.dropna(axis=0, subset=['NSC2'])  # dropping na rows when only one drug is screened
file_withNA = file[file.isnull().any(axis=1)]  # getting all the rows that have na;s

#%% let's start from choosing one drug
drug752 = file_noNA.loc[(file_noNA.NSC1 == 752) | (file_noNA.NSC2 == 752)]
drug752['NSC2'] = drug752['NSC2'].astype('int')  # dtype check needed in original read_csv

#%% group by
grouped = file_noNA.groupby(['NSC1', 'NSC2', 'CELLNAME'])

#%%
count = 0
for i,v in grouped:
    if len(v['PLATE'].unique()) >1:
        print(i,v)
        count +=1
        if count == 2:
            break
            # there seems to be no unique groups of NSC1 / NSC2 and CELLNAME that have been tested in multiple plates