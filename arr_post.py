# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

#%%
with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/good_plates.pickle',
          'rb') as handle:
    salt = pickle.load(handle)

# loading previously prepared df prepared in arr_prep.py

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/bad_plates.pickle', 'rb') as handle:
    sample = pickle.load(handle)

with open('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/weird_plates.pickle', 'rb') as handle:
    s = pickle.load(handle)