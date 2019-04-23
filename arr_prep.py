#%%
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle


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

#%%
# create dictionaries of index: value for the following columns.
# done to exchange datatype object to int
cellnames = dict(enumerate(file1.CELLNAME.unique()))
plates = dict(enumerate(file1.PLATE.unique()))
screeners = dict(enumerate(file1.SCREENER.unique()))
studies = dict(enumerate(file1.STUDY.unique()))

# inverse mapping, exchange value with key
inv_cellnames = {v: k for k, v in cellnames.items()}
inv_plates = {v: k for k, v in plates.items()}
inv_screeners = {v: k for k,v in screeners.items()}
inv_studies = {v: k for k,v in studies.items()}

# exchange columns containing objects with ints. Probably something like this can be done during reading
# reverse is done by using non inverse dictionaries with a map function, like so file1['SCREENER'] = file1['SCREENER'].map(screeners)
file1['CELLNAME'] = file1['CELLNAME'].map(inv_cellnames)
file1['PLATE'] = file1['PLATE'].map(inv_plates)
file1['SCREENER'] = file1['SCREENER'].map(inv_screeners)
file1['STUDY'] = file1['STUDY'].map(inv_studies)

# normalize the types. used to be int64 vs Int64
file1 = file1.astype({'SCREENER': 'Int64', 'STUDY': 'Int64', 'PLATE': 'Int64', 'CELLNAME': 'Int64' })

#%%
# precalc single drug mean percentgrowthnotz
# done in a separate process
hh = {}

def get_singles(x):
    if np.unique(x.NSC2) == -9999:
        hh[x.name] = np.mean(x['PERCENTGROWTHNOTZ'])
    else:
        pass

tqdm.pandas(desc="pbar")

file1.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2']).progress_apply(get_singles)

# %%
good_plates = pd.DataFrame()
bad_plates = pd.DataFrame()

gp = []  # good plates
bp = []  # bad plates
weird_plates = []  # with more than one cell line tested


def goodness_of_plate(x, bp=bp, gp=gp):
    """
    we are checkign if the plate is good.
    Good plate is the one where all combo drugs are tested as single drugs in the same plate AND there is one cell line per plate
    :param x: dataframe grouped by PLATE
    :param bp: bad plate id holder, list
    :param gp: good plate id holder, list
    :return: dataframe with mutated good plate / bad plate
    """
    if len(np.unique(x.CELLNAME)) == 1:
        a1 = x.loc[x.NSC2 != -9999][['NSC1', 'CONC1']].drop_duplicates().values  # get NSC1 unique drug - conc pairs
        a2 = x.loc[x.NSC2 != -9999][['NSC2', 'CONC2']].drop_duplicates().values  # get NSC2 unique drug - conc pairs
        a = np.concatenate(
            (a1, a2))  # concat both lists. These are all the drugs tested in combos with corresponding concentrations
        b = x.loc[x.NSC2 == -9999][
            ['NSC1', 'CONC1']].drop_duplicates().values  # all the drugs with concentrations tested as single drugs
        mask = np.isin(a, b)
        # if all true in mask,
        # it is a good plate
        if mask.all():
            gp.append(np.unique(x.PLATE))
            x['good plate'] = 1
            x['bad plate'] = 0
            return x
        else:
            bp.append(np.unique(x.PLATE))
            x['bad plate'] = 1
            x['good plate'] = 0
            return x
    else:  # this is the case for weird plate, where there is more than 1 cell line per plate
        return x



# %%
test = file1.copy()

test['mean noTZ'] = -9999
test['good plate'] = np.nan
test['bad plate'] = np.nan

tqdm.pandas(desc="progress")
test = test.groupby('PLATE').progress_apply(goodness_of_plate)

#%%

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/single_drug_meannoTZ.pickle', 'wb') as handle:
    pickle.dump(hh, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/cleaned_NCI_input.pickle', 'wb') as handle:
    pickle.dump(file1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/with_plates_sorted.pickle', 'wb') as handle:
    pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)