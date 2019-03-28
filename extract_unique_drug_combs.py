#%%
import pandas as pd
import os

#%% getting input and init'ing variables
input_all = pd.read_csv('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/summary_css_dss_ic50_synergy_smiles_study.csv', sep = ';')
drugs = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drug-2.xlsx')

#%%
# this creates a list of uniqe drug_row and drug_col combinations
grouped = input_all.groupby(['drug_row', 'drug_col'])

holder = []
for key, group in grouped:
    holder.append(key)

#%%
# this step is needed to test for presence of drug combinations that are duplicated, but in reverse
# so i m iterating over the list and then reversing the order of the elements and testing for presence of that
# if present add to new replicates list
replicates = []
temp = []
for i in holder:
    a = i[0]
    b = i[1]
    c = (b,a)
    temp.append(c)
    if (c in holder) & (i not in temp):
        replicates.append(c)
#%%
# removing the list with list comprehension
# creating a dataframe and naming the columns
'''
('ANASTROZOLE', '34793-34-5') in drug_combs
Out[9]: False
('34793-34-5','ANASTROZOLE') in drug_combs
'''
drug_combs = [x for x in holder if x not in replicates]
out = pd.DataFrame(drug_combs)
out.columns = ['DrugA','DrugB']
#%%
out.to_csv(os.getcwd()+'/list_drug_combs.csv', sep=';')
