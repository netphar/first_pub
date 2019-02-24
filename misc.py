#%%
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

sns.set()
#%% reading in the data
input_all = pd.read_csv('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/summary_css_dss_ic50_synergy_smiles_study.csv', sep = ';')
cellline = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/cell_line.xlsx')
input_doses = pd.read_csv('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drugs_cell_lines_separate_doses.csv', sep = ',')


#%%
out = input_all.groupby(['study'])

#%% getting number of drugs in each of the studies
# https://stackoverflow.com/questions/26977076/pandas-unique-values-multiple-columns
for name, group in out:
    count = len(pd.unique(group[['drug_row_id', 'drug_col_id']].values.ravel('K')))
    print('for {0} number of unique drugs is {1}'.format(name, count))

#%% get number of combinations. it is pretty much a number of rows
for name, group in out:
    if name == 'ONEIL':
        temp = group
    count = len(group.index)
    print('for {0} number of unique drug combos is {1}'.format(name, count))

#%% get number of cell lines
for name, group in out:
    count = group.cell_line_id.nunique()
    print('for {0} number of unique cell lines is {1}'.format(name, count))

#%% get tissues
for name, group in out:
    count = pd.merge(cellline[['id', 'tissue_name']], group['cell_line_id'], left_on='id', right_on='cell_line_id')['tissue_name'].nunique()
    print('for {0} number of unique tissues is {1}'.format(name, count))

#%% get dose matrices
blocks = {}
for name, group in out:
    blocks[name] = group.block_id.values

for i,v in blocks.items():  # here v is the list of block_id values
    count = input_doses.loc[input_doses.block_id.isin(v)].groupby('block_id').size().drop_duplicates().values
    print('for {0} drug matrices are {1}'.format(i, count))