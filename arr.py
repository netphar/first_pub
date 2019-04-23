# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# %%
# loading previously prepared dict with 'NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2' as keys and mean no TZ as value
with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/single_drug_meannoTZ.pickle',
          'rb') as handle:
    hh = pickle.load(handle)

# loading previously prepared df prepared in arr_prep.py

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/cleaned_NCI_input.pickle', 'rb') as handle:
    file1 = pickle.load(handle)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/with_plates_sorted.pickle', 'rb') as handle:
    test = pickle.load(handle)

# %%
# let's do conditional computation for good plates
tqdm.pandas(desc="progress")


def good_mod(x):
    '''
    for good plates we simply get means from the plates for all both singles and combos.
    add zero columns
    correct concentrations to uM
    :param x: dataframe object from groupby operation on the full file with PLATE as groupby var
    :return: return df with ['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2', 'PLATE', 'mean noTZ', 'good plate', 'bad plate']
    '''
    x = x.drop(['COMBODRUGSEQ', 'SCREENER', 'STUDY', 'TESTDATE'], axis=1)
    x['mean noTZ'] = x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)
    x = x.drop(['PERCENTGROWTHNOTZ'], axis=1)


    cellline = int(x['CELLNAME'].drop_duplicates())
    plate = int(x['PLATE'].drop_duplicates())

    # creating zero columns
    a = x.loc[x.NSC2 != -9999, :]
    a = a[['NSC1', 'NSC2']].drop_duplicates()
    a['CONC1'] = 0
    a['CONC2'] = 0
    a['mean noTZ'] = 100
    a['CELLNAME'] = cellline
    a['good plate'] = 1
    a['bad plate'] = 0
    a['PLATE'] = plate

    # correcting concentrations (mult by 10e6 and make -9999 to 0)
    x.loc[x['CONC2'] == -9999, 'CONC2'] = 0
    x[['CONC1', 'CONC2']] = x[['CONC1', 'CONC2']] * 1e6

    out = pd.concat([x, a], sort=False)
    return out

apvp = test.loc[test['good plate'] == 1, :]
ew = apvp.loc[apvp['PLATE'] == 0, :]
a = good_mod(ew)
a.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/fun.csv', sep=';')


salt = apvp.groupby('PLATE').progress_apply(good_mod)


# %%
# let's do conditional computation for bad plates.
# we groupby by plates, as we dont need to care about weird plates here. Weird plates are the ones where we have more than one cellline per plate
# return a df where all the drugs that have not been tested in that plate as singles are appended to df slice with only combos tested


def bb(x, single_drug_mean_noTZ=hh):
    """
    process bad plates. Drops all single drugs tested in a plate and substitutes with values from hh
    adds zero concentrations of drugs with mean noTZ == 100
    correction concetration to uM
    :param x: input df, group by PLATE objext with bad_plate == 1
    :param single_drug_mean_noTZ: dictionary containing ('NSC1', -9999, 'CELLNAME', 'CONC1', -9999) as keys and mean noTZ as values
    :return: cleaned df with combos and single drug values taken from hh
    """
    cellline = int(x['CELLNAME'].drop_duplicates())
    plate = int(x['PLATE'].drop_duplicates())


    a = x.loc[x.NSC2 != -9999, :].drop(['COMBODRUGSEQ', 'SCREENER', 'STUDY', 'TESTDATE'],
                                       axis=1)  # remove unneeded cols
    a['mean noTZ'] = a.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(
        np.mean)  # calc mean noTZ for combos
    a = a.drop(['PERCENTGROWTHNOTZ'], axis=1)


    # extract all drugs / conc cellline tested in combos and create a new df with it
    a1 = a[['NSC1', 'CONC1', 'CELLNAME']]
    a2 = a[['NSC2', 'CONC2', 'CELLNAME']].rename(index=str, columns={"NSC2": "NSC1", "CONC2": "CONC1"})
    new = pd.concat([a1, a2]).drop_duplicates()
    new['NSC2'] = -9999
    new['CONC2'] = -9999
    new['bad plate'] = 1
    new['good plate'] = 0
    new['PLATE'] = plate

    # do row-wise calculation searching hh dict
    new['mean noTZ'] = new.apply(
        lambda row: single_drug_mean_noTZ.get((row['NSC1'], row['NSC2'], row['CELLNAME'], row['CONC1'], row['CONC2']), -9999), axis=1)

    # concat original df without single drugs and all the single drugs tested
    out = pd.concat([a, new], sort=False)

    # creating zero columns
    zeroes = a[['NSC1', 'NSC2']].drop_duplicates()
    zeroes['CONC1'] = 0
    zeroes['CONC2'] = 0
    zeroes['mean noTZ'] = 100
    zeroes['CELLNAME'] = cellline
    zeroes['PLATE'] = plate
    zeroes['good plate'] = 0
    zeroes['bad plate'] = 1

    # correcting concentrations
    out.loc[out['CONC2'] == -9999, 'CONC2'] = 0
    out[['CONC1', 'CONC2']] = out[['CONC1', 'CONC2']] * 1e6

    real_out = pd.concat([out, zeroes], sort=False)

    return real_out


tqdm.pandas(desc="progress")

play = test.loc[test['bad plate'] == 1, :]  # all the bad plates
sample = play.groupby('PLATE').progress_apply(bb)  # this is the output to be concatenated for the final results

# %%

def bb_weird(x, single_drug_mean_noTZ=hh):
    """
    process bad plates. Drops all single drugs tested in a plate and substitutes with values from hh.
    drops plates, since we have multiples plates by group
    :param x: input df, group by ['NSC1', 'NSC2', 'CELLNAME'] objext with bad_plate.isna() and good_plate.isna()
    :param single_drug_mean_noTZ: dictionary containing ('NSC1', -9999, 'CELLNAME', 'CONC1', -9999) as keys and mean noTZ as values
    :return: cleaned df with combos and single drug values from hh
    """
    a = x.loc[x.NSC2 != -9999, :]
    if a.empty:
        pass
    else:
        a = a.drop(['COMBODRUGSEQ', 'SCREENER', 'STUDY', 'TESTDATE', 'PLATE'],
                                           axis=1)  # remove unneeded cols
        a['mean noTZ'] = a.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(
        np.mean)  # calc mean noTZ for combos
        a = a.drop(['PERCENTGROWTHNOTZ'], axis=1)

        a['bad plate'] = 0
        a['good plate'] = 0

        cellline = int(a['CELLNAME'].drop_duplicates())
#         plate = int(a['PLATE'].drop_duplicates())

        # extract all drugs / conc cellline tested in combos and create a new df with it
        a1 = a[['NSC1', 'CONC1', 'CELLNAME']]
        a2 = a[['NSC2', 'CONC2', 'CELLNAME']].rename(index=str, columns={"NSC2": "NSC1", "CONC2": "CONC1"})
        new = pd.concat([a1, a2]).drop_duplicates()
        new['NSC2'] = -9999
        new['CONC2'] = -9999
        new['bad plate'] = 0
        new['good plate'] = 0
        # do row-wise calculation searching hh dict
        new['mean noTZ'] = new.apply(
            lambda row: single_drug_mean_noTZ.get((row['NSC1'], row['NSC2'], row['CELLNAME'], row['CONC1'], row['CONC2']), -9999), axis=1)

        # creating zero columns
        zeroes = a[['NSC1', 'NSC2']].drop_duplicates()
        zeroes['CONC1'] = 0
        zeroes['CONC2'] = 0
        zeroes['mean noTZ'] = 100
        zeroes['CELLNAME'] = cellline
        zeroes['good plate'] = 0
        zeroes['bad plate'] = 0

        # concat original df without single drugs and all the single drugs tested
        out = pd.concat([a, new], sort=False)

        # correcting concentrations
        out.loc[out['CONC2'] == -9999, 'CONC2'] = 0
        out[['CONC1', 'CONC2']] = out[['CONC1', 'CONC2']] * 1e6

        real_out = pd.concat([out, zeroes], sort=False)

        return real_out

# %%
weird = test.loc[(test['bad plate'].isna()) & (test['good plate'].isna()), :]  # all the wierd plates

tqdm.pandas(desc="progress")
s = weird.groupby(['NSC1', 'NSC2', 'CELLNAME']).progress_apply(bb_weird)

#%%
with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/good_plates.pickle', 'wb') as handle:
    pickle.dump(salt, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/bad_plates.pickle', 'wb') as handle:
    pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/weird_plates.pickle', 'wb') as handle:
    pickle.dump(s, handle, protocol=pickle.HIGHEST_PROTOCOL)

