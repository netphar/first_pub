import pandas as pd
from tqdm import tqdm
import pickle

'''
This script is used to calculate the sd and cv of css values for drugA/drugB/cell_line combinations.
First separation is into combinations present in single_study vs multiple_study. These are first level dict keys


A second line of analysis is comparison of sd of css values present in multiple studies for each of the 
drugA/drugB/cell_line combos vs all drugA/drugB combos
'''

#%% Let's make sure merging works properly
"""
this part of the script generates an input file (so-called summary_css_dss_ic50_synergy_smiles_study.csv)
it uses raw files given by jing
"""
summary = pd.read_csv('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/summary.csv', sep=',')
cellline = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/cell_line.xlsx')
drugs = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drug-2.xlsx')

#%% getting input and init'ing variables
input_all = pd.read_csv('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/summary_css_dss_ic50_synergy_smiles_study.csv', sep = ';')

#%% running std analysis on input_all

grouped = input_all.groupby(['drug_row_id', 'drug_col_id', 'cell_line_id'])


#%% function for getting sd and cv of dataframe
def calc(dataframe):
    """
    :param dataframe with at least the following columns ['drug_row_id', 'drug_col_id', 'cell_line_id', 'study', 'css']
    :return: standard deviation as float, coefficient of variation as float, both rounded to 4 digits and study as set
    """
    sd = round(dataframe['css'].std(), ndigits=4)
    cv = round(sd/dataframe['css'].mean(), ndigits=4)
    study = set(dataframe['study'])
    return sd, cv, study


#%%
timer = len(grouped)
with tqdm(total=timer) as pbar:
    key_holder = {}
    for key, group in grouped:
        row = key[0]
        col = key[1]
        cell = key[2]
        reversed_key = (col, row, cell)
        if key in key_holder:  # we are checking if this combo has already been tested, if yes: pass
            pass
        else:
            if reversed_key in key_holder:  # we are checking if exchanging drug_row and drug_col ID's has already
                                            # been tested for, as these combos would be identical.
                                            # If yes: pass
                pass
            else:
                df = input_all.loc[(input_all.drug_row_id == col) & (input_all.drug_col_id == row) & (
                            input_all.cell_line_id == cell)]  # exchanging drug_row and drug_col with each other
                if not df.empty:  # if NOT empty, then exchanging resulted in additional data
                    group = group.append(df)
                    sd_out, cv_out, study_out = calc(group)
                    if len(study_out) == 1:
                        key_holder[key] = {'type': 'single_study', 'study': study_out, 'sd': sd_out, 'cv': cv_out}

                    elif len(study_out) > 1:
                        key_holder[key] = {'type': 'multiple_study', 'study': study_out, 'sd_cell_line': sd_out,
                                           'cv_cell_line': cv_out, 'study_wide': {}}
                        for study in study_out:  # if multiple studies, test drug_row and drug_col in each study
                                                 # where this combo was found across all the cell lines
                            df = input_all.loc[(input_all.drug_row_id == row) & (input_all.drug_col_id == col) & (
                                        input_all.study == study)]
                            df1 = input_all.loc[(input_all.drug_row_id == col) & (input_all.drug_col_id == row) & (
                                        input_all.study == study)]
                            df = df.append(df1)
                            sd_out_all, cv_out_all, _ = calc(df)  # we don't need study returned, as we iterate by study
                            key_holder[key]['study_wide'][study] = {'sd_study_wide': sd_out_all, 'cv_study_wide': cv_out_all}

                else:
                    if len(group.index) < 2:  # we must have more than one row to calculate sd and cv. Check for that
                        pass
                    else:
                        sd_out, cv_out, study_out = calc(group)
                        if len(study_out) == 1:
                            key_holder[key] = {'type': 'single_study', 'study': study_out, 'sd': sd_out, 'cv': cv_out}

                        elif len(study_out) > 1:
                            key_holder[key] = {'type': 'multiple_study', 'study': study_out, 'sd_cell_line': sd_out,
                                               'cv_cell_line': cv_out, 'study_wide': {}}
                            for study in study_out:
                                df = input_all.loc[(input_all.drug_row_id == row) & (input_all.drug_col_id == col) & (
                                            input_all.study == study)]
                                df1 = input_all.loc[(input_all.drug_row_id == col) & (input_all.drug_col_id == row) & (
                                            input_all.study == study)]
                                df = df.append(df1)
                                sd_out_all, cv_out_all, _ = calc(df)
                                key_holder[key]['study_wide'][study] = {'sd_study_wide': sd_out_all, 'cv_study_wide': cv_out_all}
        pbar.update(1)

# saving on disk
# with open('filter_input.pickle', 'wb') as f:
#     pickle.dump(key_holder, f, pickle.HIGHEST_PROTOCOL)

#%%
with open('/Users/zagidull/PycharmProjects/test_scientific/filter_input.pickle', 'rb') as f:
    x = pickle.load(f)

#%%
holder = {}
for i,v in x.items():
    if v['type'] == 'multiple_study' and v['study'] == {'ALMANAC', 'ONEIL'}:
        holder[i] = v['sd']
#%%
sd = []
for i,v in holder.items():
    sd.append(v)

#%%
count = 0
count1 = 0
count2 = 0
mult = {} # this holds 604 combos which are replicated across studies
oneil = {}
alma = {}
for i,v in x.items():
    if v['type'] == 'multiple_study' and v['study'] == {'ALMANAC', 'ONEIL'}:
        mult[i] = v['sd_cell_line']
        count += 1
    if v['type'] == 'single_study' and v['study'] == {'ONEIL'}:
        oneil[i] = v['sd']
        count1 += 1
    if v['type'] == 'single_study' and v['study'] == {'ALMANAC'}:
        alma[i] = v['sd']
        count2 += 1
print(count, count1, count2)
#%%
mult1 = []
oneil1 = []
alma1 = []
for i,v in mult.items():
    mult1.append(v)
for i,v in oneil.items():
    oneil1.append(v)
for i,v in alma.items():
    alma1.append(v)

#%% wilcoxon sum ranked test
# scipy.stats.wilcoxon(oneil1, mult1) # cannot be performed with unequal N in wilcoxon, so use

from scipy.stats import mannwhitneyu
mannwhitneyu(oneil1, mult1)

#%%
count = 0
oneil = {}
almanac = {}
for i,v in x.items():
    if v['type'] == 'multiple_study' and v['study'] == {'ALMANAC', 'ONEIL'}:
        almanac[i] = v['study_wide']['ALMANAC']['sd_study_wide']
        oneil[i] = v['study_wide']['ONEIL']['sd_study_wide']
        # if ['study_wide'] == 'ONEIL':
        #     oneil[i] = v['study_wide']['ONEIL']['sd_study_wide']
        print(i,v)
        count += 1
print(count)


#%%
oneil_sd = []
almanac_sd = []
for i,v in oneil.items():
    oneil_sd.append(v)
for i,v in almanac.items():
    almanac_sd.append(v)

#%%
oneil_sd = list(set(oneil_sd))
almanac_sd = list(set(almanac_sd))

# %% for this part of the analyis we take a combination from D1-D2-Cell, get rid of a cell lines and then choose
# randomly from all the cell lines present for a given combination of drugs. Thus we are making a negative control
np.random.seed(3)
count = 0
negative_control = {}
for i,v in mult.items():
    drugA = i[0]
    drugB = i[1]
    notCell = i[2]
    out_alm = input_all.loc[(input_all.drug_row_id == drugA) & (input_all.drug_col_id == drugB) &
                            (input_all.cell_line_id != notCell) & (input_all.study == 'ALMANAC')
                            ]
    out_one = input_all.loc[(input_all.drug_row_id == drugA) & (input_all.drug_col_id == drugB) &
                            (input_all.cell_line_id != notCell) & (input_all.study == 'ONEIL')
                            ]
#    print(np.append(out_alm, out_one).std())
    if not out_alm.empty:
        out_alm = round(out_alm.sample(n=1).css.iloc[0], 4)
    else:
        out_alm = round(input_all.loc[(input_all.drug_row_id == drugB) & (input_all.drug_col_id == drugA) &
                            (input_all.cell_line_id != notCell) & (input_all.study == 'ALMANAC')
                            ].css.iloc[0], 4)
    if not out_one.empty:
        out_one = round(out_one.sample(n=1).css.iloc[0], 4)
    else:
        out_one = round(input_all.loc[(input_all.drug_row_id == drugB) & (input_all.drug_col_id == drugA) &
                            (input_all.cell_line_id != notCell) & (input_all.study == 'ONEIL')
                            ].css.iloc[0], 4)

    result = round(np.std([out_alm,out_one]), 4)
    negative_control[i] = [out_alm,out_one,result]
    # count += 1
    # if count == 1:
    #     break
 #   negative_control.append(np.append(out_alm, out_one).std())

#%%
holder = []
for i,v in negative_control.items():
    holder.append(v[2])
#%%
holder_actual = []
for i,v in mult.items():
    holder_actual.append(v)

#%% wilcoxon pair test
from  scipy.stats import wilcoxon
print(wilcoxon(holder,holder_actual))

