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
