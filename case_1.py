


"""
update on case-2 (inter and intra study variability)
to check doses present in different those drugA/drugB/cell_line combos with highest SD
The idea is that css, as well as ic50 is doses dependent. If different studies used the same combo but in
very different doses - that is a problem and that results in those super high sd values.
"""
import pandas as pd

#%% first we need to load the files
summary = pd.read_csv('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/summary.csv', sep=',')
cellline = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/cell_line.xlsx')
drugs = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drug-2.xlsx')