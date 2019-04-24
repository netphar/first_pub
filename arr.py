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

# with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/cleaned_NCI_input_using_median.pickle', 'rb') as handle:
#     file1 = pickle.load(handle)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/with_plates_sorted_using_median.pickle', 'rb') as handle:
    test = pickle.load(handle)

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

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/weird_plates.pickle', 'wb') as handle:
    pickle.dump(s, handle, protocol=pickle.HIGHEST_PROTOCOL)

