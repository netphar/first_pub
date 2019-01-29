## Case 1
for descriptive analys of the dataset available in the DrugComb database is done using 

## Case 2
inter vs intra-study variability.
Create an object containing the following columns
* 'block_id',
* 'drug_row_id',
* 'drug_col_id',
* 'cell_line_id',
* 'css',
* 'synergy_zip',
* 'synergy_bliss',
* 'synergy_loewe',
* 'synergy_hsa',
* 'ic50_row',
* 'ic50_col',
* 'dss_row',
* 'dss_col',
* 'cell_line_name',
* 'drug_row',
* 'drug_col',
* 'smiles_row',
* 'smiles_col',
* 'study'

search for all the examples with identical 'drug_row', 'drug_col', 'cell_line_id'. 

Calculate the following:

1) std of all cases when a combo is present only in one study;

2) std of all cases when a combo is present in multiple studies;

NB: I am also separating out examples when reversing drug_row and drug_col creates a hit. As a result we'll get two dict: 'unipositional' and 'multipositional'. Both would contain nested dictionaries: 'present_in_single' and 'present_in_multiple'

ic50s and css value are relative to max concentrations tested, so e.g. block_ids 
22669, 22670, 22671, 22672, 196578. So here we have a problem that max values tested in ONEIL are much higher than for ALMANAC, which results in vastly different css values

rank out top 25% of combos with the highest sd present in multiple studies, see if the problem is similar to the one that we had for 


*check out whether problematic drug pairs contain a subset of concentrations, so that we can be using some sort of a scaling solution* 
## Case 3
Basic predictive modelling

