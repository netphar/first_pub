#%%
import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import category_scatter
from collections import defaultdict
import seaborn as sns
from collections import Counter
from tqdm import tqdm

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
def sorter_mod(x):
    '''
    :param x: dataframe object from groupby operation on the full file with PLATE as groupby var
    :return: dataframe with one column (labeled 0), contains values either from hh dict (for single drugs screened in bad plates or calculates mean PERCENTGROWTHNOTZ of the current dataframe
    '''
    # a,b,c,d,e = x['NSC1'].unique().astype(int).item(), x['NSC2'].unique().astype(int).item(), x['CELLNAME'].unique().astype(str).item(), x['CONC1'].unique().astype(float).item(), x['CONC2'].unique().astype(str).item()
    a = np.unique(x['NSC1'].to_numpy()).item()
    b = np.unique(x['NSC2'].to_numpy()).item()
    c = np.unique(x['CELLNAME'].to_numpy()).item()
    d = np.unique(x['CONC1'].to_numpy()).item()
    e = np.unique(x['CONC2'].to_numpy()).item()
    if b == -9999:
        #temp = np.mean(file1.loc[ (file1.NSC1 == a) & (file1.NSC2 == b) & (file1.CELLNAME == c) & (file1.CONC1 == d) & (file1.CONC2 == e)]['PERCENTGROWTHNOTZ'])
        data = {0: hh.get((a,b,c,d,e), -9999) } # this is a precalculated dictionary
        return pd.DataFrame(index=x.index, data = data)
    else:
        data = {0: np.mean(x['PERCENTGROWTHNOTZ'])}
        return pd.DataFrame(index=x.index, data=data)

def good(x):
    '''
    for good plates we simply get means from the plates
    :param x: dataframe object from groupby operation on the full file with PLATE as groupby var
    :return: transform takes in columns as series, and returns series of the same length / index with the function applied
    '''
    return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)

def bad_mod(x):
    '''
    for bad plates we need to get averages of all the screens for that combination of drugs / cell lines / doses
    we are using sorter_mod() and applying it group-wise
    :param x: dataframe object from groupby operation on the full file with PLATE as groupby var
    :return:
    '''
    return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2']).apply(sorter_mod)

hh = {}
grouped = file1.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])

for i,v in grouped:
    if i[1] == -9999:
        print(i)
        hh[i] = np.mean(v['PERCENTGROWTHNOTZ'])

del grouped

#%% good plates. Ie the ones where single drug is tested in the same plate as a combination should be analyzed using single drug values from that plate only
# for bad plates, ie the ones where single drug is tested on a different plate than the combination are analyzed such that:
#  we took the median average on the single drugs and then combined with combo data
#bad_plate_ids = list(set(bad_plate_holder))
#good_plate_ids = list(set(good_plate_holder))

# len(set(good_plate_ids).intersection(set(bad_plate_ids))) equals to 2538. Which means that these many plates are overlapping

# let's know get difference between sets. So only plates that are present in good_plate_ids
#difference = list(set(good_plate_ids) - set(bad_plate_ids))  # len = 19605

# let's use a plate CD071CP105BT549 as an example, which is difference[0]
#test = file1.loc[file1.PLATE == difference[0]]

#test.groupby(['NSC1', 'NSC2'])['PERCENTGROWTHNOTZ'].agg(np.mean)

# a good plate is such that every single drug screened in combos is also screened as a monotherapy in the same plate
#combos = np.unique(test.loc[test.NSC2 != -9999][['NSC1','NSC2']].values.flatten())  # get's a list of drugs used in comb therapy
                                                                           # NB: we need to check for the nature of the cell line as well
#singles = np.unique(test.loc[test.NSC2 == -9999][['NSC1','NSC2']].values.flatten())

# [x for x in singles if x not in combos]  # present only in b
# [x for x in combos if x not in singles]  # present only in a

file1_grouped_plate = file1.groupby(['PLATE'])
something_good = pd.DataFrame()  # holder for all good plates processed
something_bad = pd.DataFrame()  # holder for all good plates processed
something_weird = pd.DataFrame()



gp = []  # good plates
bp = []  # bad plates
weird_plates = [] # with more than one cell line tested
for i,v in file1_grouped_plate:
    if len(v['CELLNAME'].unique()) == 1:
        a1 = v.loc[v.NSC2 != -9999][
            ['NSC1', 'CONC1']].drop_duplicates().values  # get NSC1 unique drug - conc pairs
        a2 = v.loc[v.NSC2 != -9999][
            ['NSC2', 'CONC2']].drop_duplicates().values  # get NSC2 unique drug - conc pairs
        a = np.concatenate(
            (a1, a2))  # concat both lists. These are all the drugs tested in combos with corresponding concentrations
        b = v.loc[v.NSC2 == -9999][
            ['NSC1', 'CONC1']].drop_duplicates().values  # all the drugs with concentrations tested as single drugs
        mask = np.isin(a, b)  # check for presence of elements in a in b. Good plates should have all combo drugs tested as singles in the same plate
        if mask.all():  # means all drugs tested combos are tested in singles in  that very plate. Total of 15103
            temp = good(v)
            something_good = pd.concat([something_good, temp])
            gp.append(i)
            print('good plate')
        else: # total of 7124
            temp = bad_mod(v)
            something_bad = pd.concat([something_bad, temp])
            bp.append(i)
            print('bad plate')
        # below is old
        # combos = v.loc[v.NSC2 != -9999][['NSC1', 'NSC2', 'CELLNAME']].drop_duplicates()  # combos
        # singles = v.loc[v.NSC2 == -9999][['NSC1', 'NSC2', 'CELLNAME']].drop_duplicates()  # singles
        # combos_cell = set()
        # singles_cell = set()
        # combos.apply(lambda x: combos_cell.add((x[0], x[2])), axis=1)  # adding NSC1-cell line, iterating over rows
        # combos.apply(lambda x: combos_cell.add((x[1], x[2])), axis=1)  # adding NSC2-cell line, iterate over rows
        # singles.apply(lambda x: singles_cell.add((x[0], x[2])), axis=1)  # adding NSC1 from single drugs, iter over rows
        # temp = combos_cell - singles_cell  # difference should be zero if all single drugs are screened in the same plate as combo

    else: # apparently there is tons of plates with more than one cell line grown per plate. total of 4460
        temp = 

        print(i)
        weird_plates.append(i)

something_good.rename(columns={0:'mean noTZ'},inplace=True)
something_bad.rename(columns={0:'mean noTZ'},inplace=True)


test = file1.copy()
test['mean noTZ'] = np.nan
test['good plate'] = np.nan
test.update(something_good)
test_update(something_bad)


#%%
# omg, it s so much easier to just pre-calculate the values for all possible combinations and then do the look up
# fuck
counter = 0
hh = {}
grouped = file1.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])

for i,v in grouped:
    print(i)
    if i[1] == -9999:
        print(i)
        hh[i] = np.mean(v['PERCENTGROWTHNOTZ'])

del grouped
del file1_grouped_plate
#%%

test = file1.copy()
# test = file1.loc[(file1.PLATE == bp[0]) | (file1.PLATE == gp[0]) |
#                  (file1.PLATE == gp[34]) | (file1.PLATE == gp[150]) |
#                  (file1.PLATE == gp[45]) | (file1.PLATE == gp[0]) |
#                  (file1.PLATE == bp[2]) | (file1.PLATE == bp[75]) |
#                  (file1.PLATE == gp[45]) | (file1.PLATE == bp[888]) |
#                  (file1.PLATE == gp[1000])]
test['mean noTZ'] = np.nan
test['good plate'] = np.nan

#%%
def sorter(x):
    '''
    :param x: dataframe object from groupby operation on the full file with PLATE as groupby var
    :return: dataframe with one column (labeled 0), contains values either from hh dict (for single drugs screened in bad plates or calculates mean PERCENTGROWTHNOTZ of the current dataframe
    '''
    # a,b,c,d,e = x['NSC1'].unique().astype(int).item(), x['NSC2'].unique().astype(int).item(), x['CELLNAME'].unique().astype(str).item(), x['CONC1'].unique().astype(float).item(), x['CONC2'].unique().astype(str).item()
    a = np.unique(x['NSC1'].to_numpy()).item()
    b = np.unique(x['NSC2'].to_numpy()).item()
    c = x['CELLNAME'].unique().item()
    d = x['CONC1'].unique().item()
    e = x['CONC2'].unique().item()
    if b == -9999:
        #temp = np.mean(file1.loc[ (file1.NSC1 == a) & (file1.NSC2 == b) & (file1.CELLNAME == c) & (file1.CONC1 == d) & (file1.CONC2 == e)]['PERCENTGROWTHNOTZ'])
        data = {0: hh.get((a,b,c,d,e), -9999) }
        return pd.DataFrame(index=x.index, data = data)
    else:
        data = {0: np.mean(x['PERCENTGROWTHNOTZ'])}
        return pd.DataFrame(index=x.index, data=data)
    # else:
    #     return np.mean(x['PERCENTGROWTHNOTZ'])


    # if x.NSC2 == -9999:
    #     x.PERCENTGROWTHNOTZ = -1
    #     return x
    # else:
    #     x.PERCENTGROWTHNOTZ = 1
    #     return x

def good(x):
    '''
    for good plates we simply get means from the plates
    :param x: dataframe object from groupby operation on the full file with PLATE as groupby var
    :return: transform takes in columns as series, and returns series of the same length / index with the function applied
    '''
    return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)

def bad(x):
    '''
    for bad plates we need to get averages of all the screens for that combination of drugs / cell lines / doses
    we are using sorter() and applying it group-wise
    :param x: dataframe object from groupby operation on the full file with PLATE as groupby var
    :return:
    '''
    return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2']).apply(sorter)
    # for i,v in temp:
    #     hh.get[]

    # return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2']).transform(sorter)
    #
    # if x.NSC2 == -9999:
    #     ap = file1.loc[(file1.NSC1 == x.NSC1) &
    #                    (file1.NSC2 == x.NSC2) &
    #                    (file1.CELLNAME == x.CELLNAME) &
    #                    (file1.CONC1 == x.CONC1) &
    #                    (file1.CONC2 == x.CONC2)]['PERCENTGROWTHNOTZ']
    #     x['mean noTZ'] = hh.get((x.NSC1, x.NSC2, x.CELLNAME, x.CONC1,x.CONC2), -9999)
    #
    #     # x['mean noTZ'] = np.mean(file1.loc[(file1.NSC1 == x.NSC1) &
    #     #                                    (file1.NSC2 == -9999) &
    #     #                                    (file1.CELLNAME == x.CELLNAME) &
    #     #                                    (file1.CONC1 == x.CONC1) &
    #     #                                    (file1.CONC2 == -9999)]['PERCENTGROWTHNOTZ'])
    # else:
    #     # temp = file1.loc[file1.PLATE == i]
    #     # x['mean noTZ'] = np.mean(v.loc[(v.NSC1 == x.NSC1) &
    #     #                                    (v.NSC2 == x.NSC2) &
    #     #                                    (v.CELLNAME == x.CELLNAME) &
    #     #                                    (v.CONC1 == x.CONC1) &
    #     #                                    (v.CONC2 == x.CONC2)]['PERCENTGROWTHNOTZ'])
    #     return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)
    # # return x



#     temp = list(x.loc[x.NSC2 == -9999].groupby(['NSC1', 'CELLNAME', 'CONC1']).groups.keys())
#     di = {}
#     for i in temp:
#         NSC1,CELLNAME, CONC1 = i[0],i[1],i[2]
#         ff = np.mean(file1.loc[(file1.NSC1 == NSC1) & (file1.NSC2 == -9999) &
#                       (file1.CELLNAME == CELLNAME) &
#                       (file1.CONC1 == CONC1) & (file1.CONC2 == -9999)]['PERCENTGROWTHNOTZ'])
# #        x.loc[(x.NSC1 == NSC1) & (x.CELLNAME == CELLNAME) & (x.CONC1 == CONC1)]['mean noTZ'] = ff
# #            ff.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)
#         di[i] = ff
#     return di


t = test.groupby('PLATE')
something = pd.DataFrame()
something1 = pd.DataFrame()
for (i,v) in t:
    print(i)
    if i in gp:
        temp = good(v)
        something = pd.concat([something, temp])
 #       v['mean noTZ'] = temp
#        test.update(v) how about we do a bulk update. Here we just concat to create a longass pd.Series

    elif i in bp:
        temp = bad(v)
        something1 = pd.concat([something1, temp])
#        test.update(v)
something.rename(columns={0:'mean noTZ'},inplace=True)
something1.rename(columns={0:'mean noTZ'},inplace=True)

test.update(something)
test.update(something1)

test['good plate'].loc[test.PLATE.isin(gp)] = 1
test['good plate'].loc[test.PLATE.isin(bp)] = 0

test.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_updated_noTZ.csv', sep=',')
#########################################
# end of working script for calculating averages
#########################################

#########################################
# for reshaping and graphing part of the script
#########################################
#%%
dtypes = {'SCREENER': 'object', 'STUDY': 'object', 'TESTDATE': 'str', 'PLATE': 'object', 'NSC1': 'Int64', 'NSC2': 'Int64', 'CELLNAME' : 'object'}
test = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_updated_noTZ.csv', sep=',', dtype = dtypes, low_memory=False)
test.drop(columns=['Unnamed: 0'], inplace=True)

#%%
# this part makes is identical to the r file

def exchanger(x):  # returns key with lowest value (ie which drug is least prevalent in the column)
    d = dict(Counter(x))
    return min(d, key=lambda k: d[k])
def filler(x):  # returns key with highest value (ie which drug is more prevalent in the column)
    d = dict(Counter(x))
    return max(d, key=lambda k: d[k])



def exchanger1(x):  # returns key with lowest value (ie which drug is least prevalent in the column)
    d = dict(Counter(x))
    return min(d, key=d.get)
def filler1(x):  # returns key with highest value (ie which drug is more prevalent in the column)
    d = dict(Counter(x))
    return max(d, key=d.get)

to_keep = ['PLATE', 'NSC1', 'CONC1','NSC2', 'CONC2','PERCENTGROWTHNOTZ','VALID','CELLNAME', 'mean noTZ', 'good plate']
test = test[to_keep]

#### let's construct a dict from unique values in columns
cellnames = dict(enumerate(test.CELLNAME.unique()))
plates = dict(enumerate(test.PLATE.unique()))
inv_cellnames = {v: k for k, v in cellnames.items()}  # inverse mapping
inv_plates = {v: k for k, v in plates.items()}
test['CELLNAME'] = test['CELLNAME'].map(inv_cellnames)
test['PLATE'] = test['PLATE'].map(inv_plates)

#%%
safe = test.copy()
# test = safe.copy()
tqdm.pandas(desc="my bar!")

def func(m):
    i = m.name
    x,y,z = i[1],i[0],i[2]
    if (x != -9999) &  (y != -9999):
        a = m  # nsc2 = row, nsc1 = col october[((october$drug_row == 3088) & (october$drug_col == 752) & (october$cell_line_name == 'NCI/ADR-RES')),]

        c1 = a.CONC1.drop_duplicates()
        c2 = a.CONC2.drop_duplicates()

        if a['good plate'].unique() == 1:
            it = a.PLATE.unique().item()
            single_nsc1 = test.loc[(test.PLATE == it) & (test.NSC1 == y) & (test.NSC2 == -9999)]
            single_nsc2 = test.loc[(test.PLATE == it) & (test.NSC1 == x) & (test.NSC2 == -9999)]
        else:
            single_nsc1 = test.loc[
                (test.CELLNAME == z) & (test.NSC1 == y) & (test.NSC2 == -9999) & (test.CONC1.isin(c1)) & (
                            test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
            single_nsc2 = test.loc[
                (test.CELLNAME == z) & (test.NSC1 == x) & (test.NSC2 == -9999) & (test.CONC1.isin(c2)) & (
                            test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
        # single_nsc1.
        # single_nsc2.

        sample = pd.concat([a, single_nsc1, single_nsc2])
        sample['CONC2'].loc[sample['CONC2'] == -9999] = 0
        sample.loc[:, ['CONC1', 'CONC2']] = sample.loc[:, ['CONC1', 'CONC2']] * 1e6  # correcting to uM

        ex = exchanger1(sample.NSC1)
        sample.loc[sample['NSC1'] == ex, ['NSC1', 'NSC2', 'CONC1', 'CONC2']] = sample.loc[
            sample['NSC1'] == ex, ['NSC2', 'NSC1', 'CONC2', 'CONC1']].values
        sample['NSC1'] = filler1(sample['NSC1'])
        sample['NSC2'] = filler1(sample['NSC2'])
        sample = sample.append(sample.iloc[-1, :], ignore_index=True)
        sample.loc[sample.index[-1], ['CONC1', 'CONC2', 'PERCENTGROWTHNOTZ', 'mean noTZ']] = pd.Series(
            {'CONC1': 0, 'CONC2': 0, 'PERCENTGROWTHNOTZ': 100, 'mean noTZ': 100})
 #       sample['blockID'] = str(counter) + ':' + cellnames[sample.CELLNAME.unique().item()] implement counter
        return sample




done = test.groupby(['NSC1','NSC2', 'CELLNAME']).progress_apply(func)
done.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_reshaped.csv', sep=',', index=False)  # save for R loading


done = done.reset_index(drop=True)  # inded used to be something weird, now it is numbers


smth = done.groupby(['NSC1','NSC2', 'CELLNAME'])
counter = 0
blo = []
for i,v in smth:
    counter +=1
    s = np.full((len(v), ), counter)
    blo.append(s)  # these are indices groups in practice
tre = np.concatenate(blo)
done['blockID'] = tre

# let's substitute cellnames and plates
done['CELLNAME'] = done['CELLNAME'].map(cellnames)
done['PLATE'] = done['PLATE'].map(plates)

done['sep'] = np.full((len(done), ), ':')

done['new'] = done['blockID'].astype('str') + done['sep'] + done['CELLNAME']

done.drop(columns=['blockID', 'sep'], inplace=True)


# renaming to fit R's required format
done.rename(columns={'new' : 'blockID', 'mean noTZ': 'response','NSC2':'drug_row', 'CONC2':'conc_r', 'NSC1':'drug_col', 'CONC1':'conc_c','CELLNAME':'cell_line_name'},inplace=True)
done.drop(columns=['PERCENTGROWTHNOTZ', 'PLATE'], inplace=True)
done['conc_r_unit'] = done['conc_c_unit'] = 'uM'
done.drop(columns=['VALID'], inplace=True)
done.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_final1.csv', sep=',', index=False)  # save for R loading


def z(x):
    val = x.index.values[0].astype('str') + ':' + x['cell_line_name'].unique().item()
    x['block_id'] = val
    return x

tqdm.pandas(desc="my bar!")

hahah = done.groupby(['drug_row', 'drug_col', 'cell_line_name']).progress_apply(z)
hahah.drop(columns=['blockID'], inplace=True)
hahah.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_final1.csv', sep=',', index=False)  # save for R loading



t1 = done.loc[ (done.drug_row == 279836) & (done.drug_col == 740) & (done.cell_line_name == 'ACHN')]
t2 = done.loc[ (done.drug_col == 279836) & (done.drug_row == 740) & (done.cell_line_name == 'ACHN')]

#%% for  testing func above
counter = 0
groups = []
for i,v in test.groupby(['NSC1','NSC2', 'CELLNAME']):
    x,y,z = i[1],i[0],i[2]
    if (x != -9999) &  (y != -9999):
        groups.append(i)
        print(i)
        counter +=1
        if counter == 2:
            break

hah = test.loc[(test.NSC1 == 740) & (test.NSC2 == 752)& (test.CELLNAME.isin([0,1]))]

#########
# for graphing standard error of measurement of single drug noTZ values
#########


# i think above is the fastest working function i was able to make
#%%
grouped1 = test.groupby(['NSC1', 'NSC2'])
d1 = defaultdict(list)  # using defaultdict allows to use .append method to add values to a key

for i,v in grouped1:
    if i[1] == -9999:
        temp = v.groupby(['CELLNAME', 'CONC1'])


#        temp = v.groupby(['CONC1'])
        temp1 = []
        for _,k in temp:
            temp1.append(k['PERCENTGROWTHNOTZ'].std(ddof=1))
        d1[i[0]].append(temp1)


dd1 = pd.DataFrame.from_dict(d1, orient='index', columns=['sd'])
dd1.reset_index(level=0, inplace=True)
dd1.rename(index=str, columns={"index": "drug"}, inplace=True)
dd1 = dd1['sd'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('sd').join(dd1[['drug']], how='left')

incase = dd1.copy()
#%% used for drawing a scatter with error bars
from scipy import stats

dd1 = incase.copy()
dd2= dd1.groupby('drug')['sd'].agg([stats.sem, np.mean])
dd2.sort_values('mean', inplace=True)

x = dd2.index.astype('str')
y = np.array(dd2['mean'])
yerr = np.array(dd2['sem'])

sns.set()
sns.set_style("whitegrid")
sns.set_context('paper')
fig = plt.figure(0)
fig.set_size_inches(25,50)
fig.tight_layout()


# actual plotting function. Y = x and x = y (exchanged) is necessary to plot it vertically. So xerr should be yerr
plt.errorbar(y = x, x = y,color='black', marker='X', linestyle='',elinewidth=4 ,ecolor='grey', capsize=10, xerr=yerr)
plt.xlim([0,20])
#plt.xticks(rotation=60)
plt.xlabel('SD',fontsize=66)
plt.ylabel('NCI drug IDs',fontsize=66)
plt.tick_params(axis = 'x', labelsize=56)
plt.tick_params(axis = 'y', labelsize=26)
plt.title('Standard Deviation of single drug screens',  fontdict={'fontsize': '70'})
plt.grid(False,axis='y')

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = 'batch_effect' + now +'.svg'

#fig1.savefig(filename, type='pdf')
fig.savefig(filename, format="svg")
plt.show()
plt.close(fig)

#%%
####################
####################
####################
####################
####################
####################
####################
for i in ['point']:
    print(i)
    sns.catplot(x='drug', y='sd', data=dd1,color='grey',kind=i,ci='sd', join=False, capsize=.2)
    fig = plt.gcf()
    fig.set_size_inches(20,15)
    plt.ylim([0,30])

    plt.xticks(rotation='vertical')

    plt.xlabel('NCI drug IDs',fontsize=30)
    plt.ylabel('std',fontsize=30)
    plt.tick_params(labelsize=12)
    plt.tick_params(axis = 'y', labelsize=20)

    plt.title('Standard Deviation of single drug screens',  fontdict={'fontsize': '36'})
    plt.show()




####################
#%%
final = []
out = pd.DataFrame()
counter = 0
countdown = 317899
ct = 0
grouped = test.groupby(['NSC1','NSC2', 'CELLNAME'])


for i,v in grouped:
    print(countdown - ct)
    ct += 1
    x,y,z = i[1],i[0],i[2]
    if (x != -9999) &  (y != -9999):
        counter += 1
        a = v# nsc2 = row, nsc1 = col october[((october$drug_row == 3088) & (october$drug_col == 752) & (october$cell_line_name == 'NCI/ADR-RES')),]

        c1 = a.CONC1.drop_duplicates()
        c2 = a.CONC2.drop_duplicates()

        if a['good plate'].unique() == 1:
            it = a.PLATE.unique().item()
            single_nsc1 = test.loc[(test.PLATE == it) & (test.NSC1 == y) & (test.NSC2 == -9999)]
            single_nsc2 = test.loc[(test.PLATE == it) & (test.NSC1 == x) & (test.NSC2 == -9999)]
        else:
            single_nsc1 = test.loc[(test.CELLNAME == z) & (test.NSC1 == y) & (test.NSC2 == -9999) & (test.CONC1.isin(c1)) & (test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
            single_nsc2 = test.loc[(test.CELLNAME == z) & (test.NSC1 == x) & (test.NSC2 == -9999) & (test.CONC1.isin(c2)) & (test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
           # single_nsc1.
           # single_nsc2.

        sample = pd.concat([a, single_nsc1, single_nsc2])
        sample['CONC2'].loc[sample['CONC2'] == -9999] = 0
        sample.loc[:, ['CONC1', 'CONC2']] = sample.loc[:, ['CONC1', 'CONC2']] * 1e6  # correcting to uM

        ex = exchanger1(sample.NSC1)
        sample.loc[sample['NSC1'] == ex, ['NSC1', 'NSC2', 'CONC1', 'CONC2']] = sample.loc[
            sample['NSC1'] == ex, ['NSC2', 'NSC1', 'CONC2', 'CONC1']].values
        sample['NSC1'] = filler1(sample['NSC1'])
        sample['NSC2'] = filler1(sample['NSC2'])
        sample = sample.append(sample.iloc[-1, :], ignore_index=True)
        sample.loc[sample.index[-1], ['CONC1', 'CONC2', 'PERCENTGROWTHNOTZ', 'mean noTZ']] = pd.Series(
            {'CONC1': 0, 'CONC2': 0, 'PERCENTGROWTHNOTZ': 100, 'mean noTZ': 100})
        sample['blockID'] = str(counter) + ':' + cellnames[sample.CELLNAME.unique().item()]
        final.append(sample)

        # if counter == 50:
        #     break

out = pd.concat(final)
out['CELLNAME'] = out['CELLNAME'].map(cellnames)
out['PLATE'] = out['PLATE'].map(plates)

#%%
out.to_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_transformed.csv', sep=',', index=False)  # save for R loading
#%%

#######################
# end of script for creating correct blocks
#######################

#%%

x,y,z = 3088,752,'NCI/ADR-RES'
x,y,z = 763371, 757441, 'NCI-H23'

def putter(x,y,z, counter):
    '''
    :param x: NSC2 (drug_row)
    :param y: NSC1 (drug_col)
    :param z: cell line
    :return: cleaned and prepared dataframe. With unique blockID though
    '''
    a = test.loc[(test.NSC2 == x) & (test.NSC1 == y) & (test.CELLNAME == z)]  # nsc2 = row, nsc1 = col october[((october$drug_row == 3088) & (october$drug_col == 752) & (october$cell_line_name == 'NCI/ADR-RES')),]

    c1 = a.CONC1.drop_duplicates()
    c2 = a.CONC2.drop_duplicates()

    if a['good plate'].unique() == 1:
        single_nsc1 = test.loc[(test.PLATE == a.PLATE.unique().item()) & (test.NSC1 == y) & (test.NSC2 == -9999)]
        single_nsc2 = test.loc[(test.PLATE == a.PLATE.unique().item()) & (test.NSC1 == x) & (test.NSC2 == -9999)]
    else:
        single_nsc1 = test.loc[(test.CELLNAME == z) & (test.NSC1 == y) & (test.NSC2 == -9999) & (test.CONC1.isin(c1)) & (test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
        single_nsc2 = test.loc[(test.CELLNAME == z) & (test.NSC1 == x) & (test.NSC2 == -9999) & (test.CONC1.isin(c2)) & (test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])



    sample = pd.concat([a,single_nsc1,single_nsc2])
    sample['CONC2'].loc[sample['CONC2'] == -9999] = 0
    sample.loc[:,['CONC1','CONC2']] = sample.loc[:,['CONC1','CONC2']]*1e6  # correcting to uM



    ex = exchanger(sample.NSC1)
    sample.loc[sample['NSC1'] == ex, ['NSC1', 'NSC2', 'CONC1', 'CONC2']] = sample.loc[sample['NSC1'] == ex, ['NSC2', 'NSC1', 'CONC2', 'CONC1']].values
    sample['NSC1'] = filler(sample['NSC1'])
    sample['NSC2'] = filler(sample['NSC2'])
    sample = sample.append(sample.iloc[-1,:], ignore_index=True)
    sample.loc[sample.index[-1], ['CONC1','CONC2', 'PERCENTGROWTHNOTZ', 'mean noTZ']] = pd.Series({'CONC1':0,'CONC2':0, 'PERCENTGROWTHNOTZ':100, 'mean noTZ':100})
    sample['blockID'] = str(counter)+':'+ sample.CELLNAME.unique().item()
    return sample
rr = putter(x,y,z,0)
#d1 = test.loc[(test.NSC1 == 752) & (test.NSC2 == -9999) & (test.CELLNAME == 'NCI/ADR-RES') & (test.CONC1.isin(c1)) & (test.CONC2 == -9999)]
#d2 = test.loc[(test.NSC1 == 3088) & (test.NSC2 == -9999) & (test.CELLNAME == 'NCI/ADR-RES') & (test.CONC1.isin(c2)) & (test.CONC2 == -9999)]
#%%
final = pd.DataFrame()
grouped = test.groupby(['NSC1','NSC2', 'CELLNAME'])
counter = 0
countdown = 317899
ct = 0
for i,v in grouped:
    ct += 1
    x = i[1]
    y = i[0]
    z = i[2]
    print(countdown - ct)
    if (x != -9999) &  (y != -9999):
        counter += 1
        final = pd.concat([final,putter(x,y,z, counter)])
#        print(counter)
        # if counter == 10:
        #     break
    else:
        pass
#for i,v in holder.items():
#    final = pd.concat([final,v])
#%% concatting
from functools import partial, reduce

#%%
safe = test.copy()
# test = safe.copy()
tqdm.pandas(desc="my bar!")

def func(m):
    i = m.name
    x,y,z = i[1],i[0],i[2]
    if (x != -9999) &  (y != -9999):
        a = m  # nsc2 = row, nsc1 = col october[((october$drug_row == 3088) & (october$drug_col == 752) & (october$cell_line_name == 'NCI/ADR-RES')),]

        c1 = a.CONC1.drop_duplicates()
        c2 = a.CONC2.drop_duplicates()

        if a['good plate'].unique() == 1:
            it = a.PLATE.unique().item()
            single_nsc1 = test.loc[(test.PLATE == it) & (test.NSC1 == y) & (test.NSC2 == -9999)]
            single_nsc2 = test.loc[(test.PLATE == it) & (test.NSC1 == x) & (test.NSC2 == -9999)]
        else:
            single_nsc1 = test.loc[
                (test.CELLNAME == z) & (test.NSC1 == y) & (test.NSC2 == -9999) & (test.CONC1.isin(c1)) & (
                            test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
            single_nsc2 = test.loc[
                (test.CELLNAME == z) & (test.NSC1 == x) & (test.NSC2 == -9999) & (test.CONC1.isin(c2)) & (
                            test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
        # single_nsc1.
        # single_nsc2.

        sample = pd.concat([a, single_nsc1, single_nsc2])
        sample['CONC2'].loc[sample['CONC2'] == -9999] = 0
        sample.loc[:, ['CONC1', 'CONC2']] = sample.loc[:, ['CONC1', 'CONC2']] * 1e6  # correcting to uM

        ex = exchanger1(sample.NSC1)
        sample.loc[sample['NSC1'] == ex, ['NSC1', 'NSC2', 'CONC1', 'CONC2']] = sample.loc[
            sample['NSC1'] == ex, ['NSC2', 'NSC1', 'CONC2', 'CONC1']].values
        sample['NSC1'] = filler1(sample['NSC1'])
        sample['NSC2'] = filler1(sample['NSC2'])
        sample = sample.append(sample.iloc[-1, :], ignore_index=True)
        sample.loc[sample.index[-1], ['CONC1', 'CONC2', 'PERCENTGROWTHNOTZ', 'mean noTZ']] = pd.Series(
            {'CONC1': 0, 'CONC2': 0, 'PERCENTGROWTHNOTZ': 100, 'mean noTZ': 100})
 #       sample['blockID'] = str(counter) + ':' + cellnames[sample.CELLNAME.unique().item()] implement counter
        return sample




done = test.groupby(['NSC1','NSC2', 'CELLNAME']).progress_apply(func)


counter = 0
blo = {}
for i,v in done.groupby(['NSC1','NSC2', 'CELLNAME']):
    counter +=1
    blo[i] = counter  # these are indices groups in practice


#%% for  testing func above
counter = 0
groups = []
for i,v in test.groupby(['NSC1','NSC2', 'CELLNAME']):
    x,y,z = i[1],i[0],i[2]
    if (x != -9999) &  (y != -9999):
        groups.append(i)
        print(i)
        counter +=1
        if counter == 2:
            break

hah = test.loc[(test.NSC1 == 740) & (test.NSC2 == 752)& (test.CELLNAME.isin([0,1]))]

#%%


#### let's construct a dict from unique values in columns
cellnames = dict(enumerate(test.CELLNAME.unique()))
plates = dict(enumerate(test.PLATE.unique()))
inv_cellnames = {v: k for k, v in cellnames.items()}  # inverse mapping
inv_plates = {v: k for k, v in plates.items()}
test['CELLNAME'] = test['CELLNAME'].map(inv_cellnames)
test['PLATE'] = test['PLATE'].map(inv_plates)

final = pd.DataFrame()
counter = 0
countdown = 317899
ct = 0
grouped = test.groupby(['NSC1','NSC2', 'CELLNAME'])


for i,v in grouped:
    print(countdown - ct)
    ct += 1
    x,y,z = i[1],i[0],i[2]
    if (x != -9999) &  (y != -9999):
        counter += 1
        a = v# nsc2 = row, nsc1 = col october[((october$drug_row == 3088) & (october$drug_col == 752) & (october$cell_line_name == 'NCI/ADR-RES')),]

        c1 = a.CONC1.drop_duplicates()
        c2 = a.CONC2.drop_duplicates()

        if a['good plate'].unique() == 1:
            single_nsc1 = test.loc[(test.PLATE == a.PLATE.unique().item()) & (test.NSC1 == y) & (test.NSC2 == -9999)]
            single_nsc2 = test.loc[(test.PLATE == a.PLATE.unique().item()) & (test.NSC1 == x) & (test.NSC2 == -9999)]
        else:
            single_nsc1 = test.loc[(test.CELLNAME == z) & (test.NSC1 == y) & (test.NSC2 == -9999) & (test.CONC1.isin(c1)) & (test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
            single_nsc2 = test.loc[(test.CELLNAME == z) & (test.NSC1 == x) & (test.NSC2 == -9999) & (test.CONC1.isin(c2)) & (test['good plate'] == 0)].drop_duplicates(subset=['mean noTZ'])
           # single_nsc1.
           # single_nsc2.

        sample = pd.concat([a, single_nsc1, single_nsc2])
        sample['CONC2'].loc[sample['CONC2'] == -9999] = 0
        sample.loc[:, ['CONC1', 'CONC2']] = sample.loc[:, ['CONC1', 'CONC2']] * 1e6  # correcting to uM

        ex = exchanger(sample.NSC1)
        sample.loc[sample['NSC1'] == ex, ['NSC1', 'NSC2', 'CONC1', 'CONC2']] = sample.loc[
            sample['NSC1'] == ex, ['NSC2', 'NSC1', 'CONC2', 'CONC1']].values
        sample['NSC1'] = filler(sample['NSC1'])
        sample['NSC2'] = filler(sample['NSC2'])
        sample = sample.append(sample.iloc[-1, :], ignore_index=True)
        sample.loc[sample.index[-1], ['CONC1', 'CONC2', 'PERCENTGROWTHNOTZ', 'mean noTZ']] = pd.Series(
            {'CONC1': 0, 'CONC2': 0, 'PERCENTGROWTHNOTZ': 100, 'mean noTZ': 100})
        sample['blockID'] = str(counter) + ':' + sample.CELLNAME.unique().item()
        final = pd.concat([final, sample])

        if counter == 50:
            break
#######################
# end of script for creating correct blocks
#######################
#%%
file = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017.csv', sep=',',
                   dtype=dtypes) #, parse_dates=parse_dates)
file1_with_Negative = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_imputedwith_Negative.csv', sep=',',
                   dtype=dtypes) #, parse_dates=parse_dates)
original = file.copy()  # this is a copy of the file just in case

#%%
file.fillna(pd.Series(-9999, index=file.select_dtypes(exclude='category').columns), inplace=True)
file_grouped = file.groupby(['NSC1', 'NSC2', 'CELLNAME'])
file1_with_Negative.fillna(pd.Series(-9999, index=file1.select_dtypes(exclude='category').columns), inplace=True)


#%% we are checking if any of the drug1 - drug2 - cell line are tested in more than one plate
single_plate_per_combo = 0
single_plate_per_combo_holder = []
many_plate_per_combo = 0
many_plate_per_combo_holder = []

for i,v in file_grouped:
    if i[1] != -9999:
        if len(v.PLATE.unique()) == 1:
            single_plate_per_combo += 1
            single_plate_per_combo_holder.append(i)

        else:
            many_plate_per_combo += 1
            many_plate_per_combo_holder.append(i)

#%% let's get # of plates for combos tested in multiple plates
sets = set()
for i in many_plate_per_combo_holder:
    sets.add(file_grouped.get_group(i).PLATE.unique().shape[0])

#%% we are checking how many good plates are there vs bad plates
# good plate is where drug1-drug2-cell line is tested in the same plate with a single drug
file_grouped_plate = file.groupby(['PLATE'])

good_plate = 0
good_plate_holder = []
bad_plate = 0
bad_plate_holder = []

for i,v in file_grouped_plate:
    gr = v.groupby(['NSC1', 'NSC2', 'CELLNAME'])
    k = gr.groups.keys()
    for o,_ in gr:
        if (o[0], -9999, o[2]) in k:
            good_plate_holder.append(i)
            good_plate += 1
        else:
            bad_plate += 1
            bad_plate_holder.append(i)



#%% testing on a sample dataframe with two series
#gp[23]
#Out[337]: '02_A5_B01'
#bp[23]
#Out[338]: '01_48'
test = file1.copy()
def good(x):  # for good plates we simply get means from the plates
    return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)

def bad(x):  # for bad plates we need to get averages of all the screens for that combination of drugs / cell lines / doses
    if x.NSC2 == -9999:
        x['mean noTZ'] = np.mean(file1.loc[(file1.NSC1 == x.NSC1) &
                                           (file1.NSC2 == -9999) &
                                           (file1.CELLNAME == x.CELLNAME) &
                                           (file1.CONC1 == x.CONC1) &
                                           (file1.CONC2 == -9999)]['PERCENTGROWTHNOTZ'])
    else:
        x['mean noTZ'] = np.mean(v.loc[(v.NSC1 == x.NSC1) &
                                           (v.NSC2 == x.NSC2) &
                                           (v.CELLNAME == x.CELLNAME) &
                                           (v.CONC1 == x.CONC1) &
                                           (v.CONC2 == x.CONC2)]['PERCENTGROWTHNOTZ'])
    return x



#     temp = list(x.loc[x.NSC2 == -9999].groupby(['NSC1', 'CELLNAME', 'CONC1']).groups.keys())
#     di = {}
#     for i in temp:
#         NSC1,CELLNAME, CONC1 = i[0],i[1],i[2]
#         ff = np.mean(file1.loc[(file1.NSC1 == NSC1) & (file1.NSC2 == -9999) &
#                       (file1.CELLNAME == CELLNAME) &
#                       (file1.CONC1 == CONC1) & (file1.CONC2 == -9999)]['PERCENTGROWTHNOTZ'])
# #        x.loc[(x.NSC1 == NSC1) & (x.CELLNAME == CELLNAME) & (x.CONC1 == CONC1)]['mean noTZ'] = ff
# #            ff.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)
#         di[i] = ff
#     return di


test['mean noTZ'] = np.nan
t = test.groupby('PLATE')

for (i,v) in t:
    if i in gp:
        temp = good(v)
        v['mean noTZ'] = temp
        test.update(v)
    elif i in bp:
        v = v.apply(bad, axis=1)
        test.update(v)


#%%
print('hey')




        temp = list(v.loc[v.NSC2 == -9999].groupby(['NSC1', 'CELLNAME', 'CONC1']).groups.keys())
        di = {}
        for i in temp:
            NSC1, CELLNAME, CONC1 = i[0], i[1], i[2]
            ff = np.mean(file1.loc[(file1.NSC1 == NSC1) & (file1.NSC2 == -9999) &
                                   (file1.CELLNAME == CELLNAME) &
                                   (file1.CONC1 == CONC1) & (file1.CONC2 == -9999)]['PERCENTGROWTHNOTZ'])
            #        x.loc[(x.NSC1 == NSC1) & (x.CELLNAME == CELLNAME) & (x.CONC1 == CONC1)]['mean noTZ'] = ff
            #            ff.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)
            di[i] = ff
            #val = np.mean(file1.loc[ (file1.NSC1 == NSC1) & (file1.NSC2 == -9999) & (file1.CELLNAME == CELLNAME) & (file1.CONC1 == CONC1) & (file1.CONC2 == -9999)]['PERCENTGROWTHNOTZ']) # .groupby(['NSC1','NSC2','CELLNAME','CONC1','CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)
            #v.loc[(v.NSC2 == -9999) & (v.NSC1 == NSC1) & (v.CELLNAME == CELLNAME) & (v.CONC1 == CONC1) & (v.CONC2 == -9999)]['mean noTZ'] = val
#        p = bad(v)
#        test.loc[v.index]
        v.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2']).apply(lambda x: x)
        test.update(v)




for i,v in gg:
    if i in gp:
        temp = v.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].\
            agg(np.mean).rename('mean noTZ').\
            reset_index()

    else:
        print(i)

#%%
counter = 0
for i in gp:
    a = file1.loc[file1.PLATE == i]
    a['mean noTZ'] = a.\
        groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].\
        transform(np.mean)  # calculate mean of noTZ for each combo of drug1-drug2-cellname-conc1-conc2
    '''
    agg(np.mean).\
        rename('mean noTZ').\
        reset_index()
    '''
    counter += 1
    if counter == 1:
        break

gg = file1.groupby(['PLATE'])

for i,v in gg:
    if i in gp:
        v['gp_mean_noTZ'] = v.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ']. \
            transform(np.mean)
    elif i in bp:
        v['gp_mean_noTZ'] = 0
#%%
counter = 0
for i,v in gg:
    print(i)
    counter += 1
    if counter == 1:
        break


#%% let's calculate sd of PERCENTGROWTH for single drugs per cell line per dose

grouped = file.groupby(['NSC1', 'NSC2'])
d = defaultdict(list)  # using defaultdict allows to use .append method to add values to a key

for i,v in grouped:
    if i[1] == -9999:
        temp = v.groupby(['CELLNAME', 'CONC1'])


#        temp = v.groupby(['CONC1'])
        temp1 = []
        for _,k in temp:
            temp1.append(k['PERCENTGROWTH'].std(ddof=1))
        d[i[0]].append(temp1)


#%% let's plot the stuff above. So drug on x axis and sd on y axis
dd = pd.DataFrame.from_dict(d, orient='index', columns=['sd'])
dd.reset_index(level=0, inplace=True)
dd.rename(index=str, columns={"index": "drug"}, inplace=True)
dd = dd['sd'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('sd').join(dd[['drug']], how='left')

sns.catplot(x='drug', y='sd', kind='bar', data=dd, ci='sd')
fig = plt.gcf()
fig.set_size_inches(20,15)
plt.xticks(rotation='vertical')
plt.title('sd of single drug screens')
plt.show()


#%% let's calculate sd of PERCENTGROWTHNOTZ for single drugs per cell line per dose.
# change d and grouped to d1 and grouped1 and so on
# this is with PERCENTGROWTHNOTZ imputed using r-script sharing_notebooks/cleanup/NCI_processing.R

grouped1 = file1.groupby(['NSC1', 'NSC2'])
d1 = defaultdict(list)  # using defaultdict allows to use .append method to add values to a key

for i,v in grouped1:
    if i[1] == -9999:
        temp = v.groupby(['CELLNAME', 'CONC1'])


#        temp = v.groupby(['CONC1'])
        temp1 = []
        for _,k in temp:
            temp1.append(k['PERCENTGROWTHNOTZ'].std(ddof=1))
        d1[i[0]].append(temp1)


dd1 = pd.DataFrame.from_dict(d1, orient='index', columns=['sd'])
dd1.reset_index(level=0, inplace=True)
dd1.rename(index=str, columns={"index": "drug"}, inplace=True)
dd1 = dd1['sd'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('sd').join(dd1[['drug']], how='left')

sns.catplot(x='drug', y='sd', kind='bar', data=dd1, ci='sd', )
fig = plt.gcf()
fig.set_size_inches(20,15)
plt.ylim([-50,100])
plt.xticks(rotation='vertical')
plt.title('sd of single drug screens. with PERCENTGROWTHNOTZ imputed')
plt.show()


#%% with values below zero from PERCENTGROWTHNOTZ left as is, ie without zeroing. AS they constitute ~6%


grouped1_with_Negative = file1_with_Negative.groupby(['NSC1', 'NSC2'])
d1_with_Negative = defaultdict(list)  # using defaultdict allows to use .append method to add values to a key

for i,v in grouped1_with_Negative:
    if i[1] == -9999:
        temp = v.groupby(['CELLNAME', 'CONC1'])


#        temp = v.groupby(['CONC1'])
        temp1 = []
        for _,k in temp:
            temp1.append(k['PERCENTGROWTHNOTZ'].std(ddof=1))
        d1_with_Negative[i[0]].append(temp1)


dd1_with_Negative = pd.DataFrame.from_dict(d1_with_Negative, orient='index', columns=['sd'])
dd1_with_Negative.reset_index(level=0, inplace=True)
dd1_with_Negative.rename(index=str, columns={"index": "drug"}, inplace=True)
dd1_with_Negative = dd1_with_Negative['sd'].apply(lambda x: pd.Series(x)).stack().reset_index(level=1, drop=True).to_frame('sd').join(dd1_with_Negative[['drug']], how='left')

sns.catplot(x='drug', y='sd', kind='bar', data=dd1_with_Negative, ci='sd', )
fig = plt.gcf()
fig.set_size_inches(20,15)
plt.ylim([-50,100])
plt.xticks(rotation='vertical')
plt.title('PERCENTGROWTHNOTZ imputed. Negative values are kept')
plt.show()

#%% to test how many examples are there of plates where monotherapy is not on the same plate as the combination
counter = 0
for i,v in file_grouped:
    if i[1] != -9999:
        q = file_grouped.get_group((i[0], -9999, i[2]))
        w = file_grouped.get_group((i[1], -9999, i[2]))
        print(i, q, w)
        counter += 1
        if counter == 1:
            break


#%%
test = file.loc[((file.NSC1 == 256439) & (file.NSC2 == 740) & (file.CELLNAME == 'HCT-15')) |
                ((file.NSC1 == 740) & (file.NSC2 == 256439) & (file.CELLNAME == 'HCT-15')) |
                ((file.NSC1 == 740) & (file.NSC2 == -9999) & (file.CELLNAME == 'HCT-15')) |
                ((file.NSC1 == 256439) & (file.NSC2 == -9999) & (file.CELLNAME == 'HCT-15'))]

#%%
test_grouped = test.groupby(['NSC1', 'NSC2'])

a = test_grouped.get_group((256439, 740))
b = test_grouped.get_group((256439, -9999))
c = test_grouped.get_group((740, -9999))

category_scatter(x='PLATE', y='PERCENTGROWTH', label_col='CONC1', data=b)
fig = plt.gcf()
fig.set_size_inches(15,10)
plt.xticks(rotation='vertical')
plt.title((256439, -9999))
plt.show()

category_scatter(x='PLATE', y='PERCENTGROWTH', label_col='CONC1', data=c)
fig = plt.gcf()
fig.set_size_inches(15,10)
plt.xticks(rotation='vertical')
plt.title((740, -9999))
plt.show()

d = defaultdict(list)
for i in c.CONC1.unique():
    d[i].append(c.loc[c.CONC1 == i].PERCENTGROWTH.values)

    c.loc[c.CONC1 == i][['PLATE', 'PERCENTGROWTH']]

c.groupby(['CONC1'])['PERCENTGROWTH'].aggregate(lambda x: np.std(x, ddof=0))


#%%

for i,v in test_grouped:
    print(i)
    v.PERCENTGROWTH
#%%
holder = []
counter = 0
for i,v in file_grouped:
    # holder.append(v['PLATE'].unique().values)
    # counter += 1
    if (len(v['PLATE'].unique()) > 1) & (

    ):
        print(v['PLATE'].unique())
        counter += 1


#%% testing for replcicated combos.
# It means that a combo of 'NSC1','NSC2','CELLNAME', 'CONC1', 'CONC2' should be tested more than once
counter = 0
# def test_reps(x):
#     i = x.name
#     print(i)
#     counter += 1
#     if counter == 2:
#         break

file1['nunique'] = file1.groupby(['NSC1','NSC2','CELLNAME', 'CONC1', 'CONC2'])['PLATE'].transform('nunique')
file2 = file1.loc[(file1['nunique'] >1) & (file1.NSC2 != -9999)]


asdasd = smth[smth.index.isin(['-9999'])]
smth.index = smth.index.map(unicode)

# for Python 3 (the unicode type does not exist and is replaced by str)
smth.index = smth.index.map(str)
# for i,v in grgr:
#     a,b,c,d,e = i[0], i[1],i[2],i[3],i[4]
#     if b != -9999:
#         if len(v.PLATE.drop_duplicates()) > 1:
#             counter +=1
#             print(i)
#             if counter == 1:
#                 break

#%%
grgr = file1.groupby(['NSC1','NSC2','CELLNAME', 'CONC1', 'CONC2'])
counter1 = 0
for ii,vv in grgr23:
    if ii[1] != -9999:
        counter1 +=1
        if counter1 == 1:
            break