# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

#%%
with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/good_plates.pickle',
          'rb') as handle:
    salt = pickle.load(handle)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/bad_plates.pickle', 'rb') as handle:
    sample = pickle.load(handle)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/single_drug_meannoTZ_using_median.pickle',
          'rb') as handle:
    hh = pickle.load(handle)
# with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/weird_plates.pickle', 'rb') as handle:
#     s = pickle.load(handle)

#%%
# qc
def qc(x, plate_id):
    """
    qc checks if all the correct drugs in correct concentrations and cell lines are tested
    :param x: df
    :param plate_id: int
    :return: print ok or not
    """
    if plate_id in x.PLATE:
        bad_test = x.loc[x.PLATE == plate_id, :]

        bad_drugs = bad_test.loc[(bad_test.NSC2 != -9999) & (bad_test.CONC1 != 0), :][
            ['NSC1', 'CONC1', 'NSC2', 'CONC2', 'CELLNAME']]  # these are combos
        bad_stacked = pd.concat([bad_drugs.loc[:, ['NSC1', 'CONC1', 'CELLNAME']], bad_drugs.loc[:, ['NSC2', 'CONC2', 'CELLNAME']]
                                .rename(index=str, columns={"NSC2": "NSC1", "CONC2": "CONC1"})], ignore_index=True,
                                sort=False).drop_duplicates()
        bad_drugs_single = bad_test.loc[(bad_test.NSC2 == -9999) & (bad_test.CONC1 != 0), :][['NSC1', 'CONC1', 'CELLNAME']]  # these are single

        bad_drugs_zeroes = bad_test.loc[(bad_test.CONC2 == 0) & (bad_test.CONC1 == 0), :][['NSC1', 'NSC2', 'CELLNAME']]
        bad_drugs_not_zeroes = bad_test.loc[(bad_test.NSC2 != -9999) & (bad_test.CONC1 != 0), :][
            ['NSC1', 'NSC2', 'CELLNAME']].drop_duplicates()

        if (np.isin(bad_stacked, bad_drugs_single).all()) & (np.isin(bad_drugs_zeroes, bad_drugs_not_zeroes).all()):  # evals to true, if we have all the single drugs tested in the same doses as multiple
            print('OK')
        else:
            print('not OK')
    else:
        print('plate is not found')

#%%
# qc
bad_test = sample.loc[sample.PLATE == 1, :]  # bad plate
good_test = salt.loc[salt.PLATE == 0, :]  # good plate
