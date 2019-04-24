# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# %%
# loading previously prepared dict with 'NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2' as keys and mean no TZ as value
with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/single_drug_meannoTZ_using_median.pickle',
          'rb') as handle:
    hh = pickle.load(handle)

# loading previously prepared df prepared in arr_prep.py


with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/with_plates_sorted_using_median.pickle', 'rb') as handle:
    test = pickle.load(handle)

# needed for mapping

#%%
'''
NB: if a drug is administered as a monotherapy in the screen, nan is given in the second column
we sub all nan's with -9999

imputed file is ComboDrugGrowth_Nov2017.csv where NA's in PERCENTGROWTHNOTZ were imputed in the following way:
fit a linear model to PERCENTGROWTH, use this model to fill NA in PERCENTGROWTHNOTZ
values below zero are set to zero, since this makes the format compatible with the way studies are run in FIMM and in other papers, ie ONEIL
'''
dtypes = {'COMBODRUGSEQ': 'Int64', 'SCREENER': 'object', 'STUDY': 'object',  'PLATE': 'object', 'NSC1': 'Int64','CONC1': 'float64', 'NSC2': 'Int64','CONC2': 'float64', 'CELLNAME': 'object'}
parse_dates = ['TESTDATE']
fields = ['COMBODRUGSEQ','SCREENER', 'STUDY', 'TESTDATE', 'PLATE', 'NSC1', 'CONC1', 'NSC2', 'CONC2', 'PERCENTGROWTHNOTZ', 'CELLNAME']

file1 = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_imputed.csv', sep=',',
                   dtype=dtypes, usecols=fields,  parse_dates=parse_dates)
# file1.drop(['Unnamed: 0'], inplace=True, axis=1)
file1.iloc[-1,-1] = 'SF-539'  # very last element is 'SF-539\x1a' for some reason


# also need to get na's removed from file1
file1.fillna(pd.Series(-9999, index=file1.select_dtypes(exclude='category').columns), inplace=True)
cellnames = dict(enumerate(file1.CELLNAME.unique()))
plates = dict(enumerate(file1.PLATE.unique()))

# %%
# let's do conditional computation for good plates
tqdm.pandas(desc="progress")


def good_mod(x):
    """
    analysis of good plate
    :param x: grouped by plate df
    :return: processed thing
    """

    # gotta use dict with nsc1 nsc2 cellname conc1 conc2 as key and mean noTZ as value
    # a very convoluted solution to calculation mean noTZ for single drugs
    hh1 = {}

    def get_singles_mod(x):
        if np.unique(x.NSC2) == -9999:
            hh1[x.name] = np.median(x['PERCENTGROWTHNOTZ'])
        else:
            pass

    x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2']).apply(get_singles_mod)
    x = x.loc[x.NSC2 != -9999, :]

    x['mean noTZ'] = x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.median)  # calc mean noTZ for combos
    x = x.drop(['PERCENTGROWTHNOTZ'], axis=1)

    # this part adds zero concentrations and an empty row
    def dropper_mod(x, single_drug_median_noTZ=hh1):
        a = x.drop_duplicates(subset=['NSC1', 'CONC1'])
        a['CONC2'] = 0
        a['mean noTZ'] = a.apply(
            lambda row: single_drug_median_noTZ.get((row['NSC1'], -9999, row['CELLNAME'], row['CONC1'], -9999), -9999),
            axis=1)


        b = x.drop_duplicates(subset=['NSC2', 'CONC2'])
        b['CONC1'] = 0
        b['mean noTZ'] = b.apply(
            lambda row: single_drug_median_noTZ.get((row['NSC2'], -9999, row['CELLNAME'], row['CONC2'], -9999), -9999),
            axis=1)

        # creating zero columns
        zeroes = x[['NSC1', 'NSC2']].drop_duplicates()
        zeroes['CONC1'] = 0
        zeroes['CONC2'] = 0
        zeroes['mean noTZ'] = 100
        zeroes['CELLNAME'] = x.name[2]
        zeroes['PLATE'] = x.name[3]
        zeroes['good plate'] = 1
        zeroes['bad plate'] = 0

        out = pd.concat([a, b], sort=False,ignore_index=True)
        out = pd.concat([out, zeroes], sort=False,ignore_index=True)

        return out
    x = x.drop(['COMBODRUGSEQ', 'SCREENER', 'STUDY', 'TESTDATE'],
                                       axis=1)  # remove unneeded cols


    b = x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'PLATE']).apply(dropper_mod)


    out = pd.concat([x,b], sort=False, ignore_index=True)

    out[['CONC1', 'CONC2']] *= 1e6

    return out

apvp = test.loc[test['good plate'] == 1, :]
# apvp = apvp.loc[ apvp.PLATE.isin([0,10]),: ]



salt = apvp.groupby(['PLATE'], as_index=False).progress_apply(good_mod)


# %%
# let's do conditional computation for bad plates.
# we groupby by plates, as we dont need to care about weird plates here. Weird plates are the ones where we have more than one cellline per plate
# return a df where all the drugs that have not been tested in that plate as singles are appended to df slice with only combos tested


def bb_mod(x, single_drug_median_noTZ=hh):
    """
    same as before
    :param x: grouped by plate
    :param single_drug_median_noTZ: dictionary containing nsc1 nsc2 cellname conc1 conc2 as key and mean noTZ calculated across all the plates
    :return: sorted and prepared df
    """


    def dropper(x, single_drug_median_noTZ=hh):
        a = x.drop_duplicates(subset=['NSC1', 'CONC1'])
        a['CONC2'] = 0
        a['mean noTZ'] = a.apply(
            lambda row: single_drug_median_noTZ.get((row['NSC1'], -9999, row['CELLNAME'], row['CONC1'], -9999), -9999),
            axis=1)

        b = x.drop_duplicates(subset=['NSC2', 'CONC2'])
        b['CONC1'] = 0
        b['mean noTZ'] = b.apply(
            lambda row: single_drug_median_noTZ.get((row['NSC2'], -9999, row['CELLNAME'], row['CONC2'], -9999), -9999),
            axis=1)

        # creating zero columns
        zeroes = x[['NSC1', 'NSC2']].drop_duplicates()
        zeroes['CONC1'] = 0
        zeroes['CONC2'] = 0
        zeroes['mean noTZ'] = 100
        zeroes['CELLNAME'] = x.name[2]
        zeroes['PLATE'] = x.name[3]
        zeroes['good plate'] = 0
        zeroes['bad plate'] = 1

        out = pd.concat([a, b], sort=False)
        out = pd.concat([out, zeroes], sort=False)

        return out

    a = x.loc[x.NSC2 != -9999, :].drop(['COMBODRUGSEQ', 'SCREENER', 'STUDY', 'TESTDATE'],
                                       axis=1)  # remove unneeded cols
    a['mean noTZ'] = a.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.median)  # calc mean noTZ for combos
    a = a.drop(['PERCENTGROWTHNOTZ'], axis=1)

    b = a.groupby(['NSC1', 'NSC2', 'CELLNAME', 'PLATE']).apply(dropper)

    out = pd.concat([a,b], sort=False, ignore_index=True)

    out[['CONC1', 'CONC2']] *= 1e6

    return out

tqdm.pandas(desc="progress")

play = test.loc[test['bad plate'] == 1, :]  # all the bad plates
# play = play.loc[ play.PLATE.isin([1,2]),: ]

sample = play.groupby(['PLATE'], as_index = False).progress_apply(bb_mod)  # this is the output to be concatenated for the final results

#%%
with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/good_plates.pickle', 'wb') as handle:
    pickle.dump(salt, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/bad_plates.pickle', 'wb') as handle:
    pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
# processing weird plates
def bb_weird(x, single_drug_median_noTZ=hh):
    """
    process weird plates. Drops all single drugs tested in a plate and substitutes with values from hh.
    drops plates, since we have multiples plates by group
    :param x: input df, group by ['NSC1', 'NSC2', 'CELLNAME'] objext with bad_plate.isna() and good_plate.isna()
    :param single_drug_mean_noTZ: dictionary containing ('NSC1', -9999, 'CELLNAME', 'CONC1', -9999) as keys and mean noTZ as values
    :return: cleaned df with combos and single drug values from hh
    """

    def dropper(x, single_drug_median_noTZ=hh):
        a = x.drop_duplicates(subset=['NSC1', 'CONC1'])
        a['CONC2'] = 0
        a['mean noTZ'] = a.apply(
            lambda row: single_drug_median_noTZ.get((row['NSC1'], -9999, row['CELLNAME'], row['CONC1'], -9999),
                                                    -9999),
            axis=1)

        b = x.drop_duplicates(subset=['NSC2', 'CONC2'])
        b['CONC1'] = 0
        b['mean noTZ'] = b.apply(
            lambda row: single_drug_median_noTZ.get((row['NSC2'], -9999, row['CELLNAME'], row['CONC2'], -9999),
                                                    -9999),
            axis=1)

        # creating zero columns
        zeroes = x[['NSC1', 'NSC2']].drop_duplicates()
        zeroes['CONC1'] = 0
        zeroes['CONC2'] = 0
        zeroes['mean noTZ'] = 100
        zeroes['CELLNAME'] = x.name[2]
        zeroes['PLATE'] = x.name[3]
        zeroes['good plate'] = 0
        zeroes['bad plate'] = 0

        out = pd.concat([a, b], sort=False)
        out = pd.concat([out, zeroes], sort=False)

        return out
    a = x.loc[x.NSC2 != -9999, :]
    if a.empty:  # check if the plate is only empty
        pass
    else:



        a = x.loc[x.NSC2 != -9999, :].drop(['COMBODRUGSEQ', 'SCREENER', 'STUDY', 'TESTDATE'],
                                           axis=1)  # remove unneeded cols
        a['mean noTZ'] = a.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(
            np.median)  # calc mean noTZ for combos
        a = a.drop(['PERCENTGROWTHNOTZ'], axis=1)

        b = a.groupby(['NSC1', 'NSC2', 'CELLNAME', 'PLATE']).apply(dropper)

        out = pd.concat([a, b], sort=False, ignore_index=True)

        out[['CONC1', 'CONC2']] *= 1e6

        return out

# %%
weird = test.loc[(test['bad plate'].isna()) & (test['good plate'].isna()), :]  # all the wierd plates
weird[['good plate', 'bad plate']] = 0
# weird = weird.loc[weird.PLATE.isin([22301, 22302]), :]


tqdm.pandas(desc="progress")
s = weird.groupby(['PLATE'], as_index = False).progress_apply(bb_weird)

#%%
# qc for weird plates
safety = s.copy()
s = s.drop_duplicates(subset = ['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2', 'mean noTZ'])
lengths = []
for i,v in s.groupby(['NSC1', 'NSC2', 'CELLNAME']):
    lengths.append(len(v))

lengths = np.unique(lengths)  # all 24
#%% adding block id and house keeping for input to synergyfinder

s['CELLNAME'] = s['CELLNAME'].map(cellnames)
s['PLATE'] = s['PLATE'].map(plates)

def func(x):
    x['block ID'] = str(abs(hash(x.name))) + ':' + str(x.name[2])
    return x

s = s.groupby(['NSC1', 'NSC2', 'CELLNAME']).apply(func)

s.rename(columns={'mean noTZ': 'response', 'NSC2':'drug_row', 'CONC2':'conc_r', 'NSC1':'drug_col', 'CONC1':'conc_c', 'CELLNAME':'cell_line_name'}, inplace=True)
s['conc_r_unit'] = s['conc_c_unit'] = 'uM'


with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/weird_plates_median.pickle', 'wb') as handle:
    pickle.dump(s, handle, protocol=pickle.HIGHEST_PROTOCOL)

s.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/weird_plates.csv', sep=',', index=False)  # save for R loading


#%% qc for bad plates
backup = sample.copy()

sample = sample.drop_duplicates(subset = ['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2', 'mean noTZ'])

lengths_bad = []
for i,v in sample.groupby(['NSC1', 'NSC2', 'CELLNAME']):
    lengths_bad.append(len(v))
    if (len(v) != 24) & (len(v) != 16):
        print(i)
        break
print(np.unique(lengths_bad))  # 16 and 24

#%% housekeeping for bad plates
tqdm.pandas(desc="progress")


sample['cell_line_name'] = sample['cell_line_name'].map(cellnames)
sample['PLATE'] = sample['PLATE'].map(plates)

def func(x):
    x['block ID'] = str(abs(hash(x.name))*2) + ':' + str(x.name[2])
    return x

sample = sample.groupby(['NSC1', 'NSC2', 'CELLNAME']).progress_apply(func)

sample.rename(columns={'mean noTZ': 'response', 'NSC2':'drug_row', 'CONC2':'conc_r', 'NSC1':'drug_col', 'CONC1':'conc_c', 'CELLNAME':'cell_line_name'}, inplace=True)
sample['conc_r_unit'] = sample['conc_c_unit'] = 'uM'


with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/bad_plates_median.pickle', 'wb') as handle:
    pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
sample.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/bad_plates.csv', sep=',', index=False)  # save for R loading

#%% qc for good plates

backup1 = salt.copy()

lengths_good = []
for i,v in salt.groupby(['NSC1', 'NSC2', 'CELLNAME', 'PLATE']):
    lengths_good.append(len(v))
    if (len(v) != 24) & (len(v) != 16):
        print(i)
        break
print(np.unique(lengths_good))

#%% good plates housekeeping

# salt = salt.drop_duplicates(subset = ['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2', 'mean noTZ'])

tqdm.pandas(desc="progress")


salt['CELLNAME'] = salt['CELLNAME'].map(cellnames)
salt['PLATE'] = salt['PLATE'].map(plates)

def func(x):
    x['block ID'] = str(abs(hash(x.name))*2) + ':' + str(x.name[2])
    return x

salt = salt.groupby(['NSC1', 'NSC2', 'CELLNAME', 'PLATE']).progress_apply(func)

salt.rename(columns={'mean noTZ': 'response', 'NSC2':'drug_row', 'CONC2':'conc_r', 'NSC1':'drug_col', 'CONC1':'conc_c', 'CELLNAME':'cell_line_name'}, inplace=True)
salt['conc_r_unit'] = salt['conc_c_unit'] = 'uM'


with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/good_plates_median.pickle', 'wb') as handle:
    pickle.dump(salt, handle, protocol=pickle.HIGHEST_PROTOCOL)

salt.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/good_plates.csv', sep=',', index=False)  # save for R loading
