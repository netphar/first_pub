"""
this script generates a descriptive analysis of the drugs present in drugcomb database from the following file
`select 'block_id','drug_row_id','drug_row_name','drug_col_id','drug_col_name','cell_line_name','response','conc_r','conc_c'  UNION ALL
SELECT summary.block_id,drug.id as drug_row_id,drug.dname as drug_row,d2.id as drug_col_id, d2.dname as drug_col,
cell_line.name as cell_line_name, response,conc_r,conc_c
FROM summary inner join drug on summary.drug_row_id=drug.id inner join
drug d2 on summary.drug_col_id=d2.id inner join cell_line on summary.cell_line_id=cell_line.id  inner join response on summary.block_id=response.block_id
into outfile '/var/lib/mysql-files/s4.csv' fields terminated by ',' enclosed by '"' lines terminated by '\n';`

"""
import pandas as pd

#%% first we need to load the files
summary = pd.read_csv('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drugs_cell_lines_separate_doses.csv', sep=',')
#%%