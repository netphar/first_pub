#%%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import category_scatter
from collections import defaultdict
import seaborn as sns

#%%
'''
NB: if a drug is administered as a monotherapy in the screen, nan is given in the second column
we sub all nan's with -9999
'''
dtypes = {'SCREENER': 'object', 'STUDY': 'object', 'TESTDATE': 'str', 'PLATE': 'object', 'NSC1': 'Int64', 'NSC2': 'Int64'}
# parse_dates = ['TESTDATE']

file1 = pd.read_csv('/Users/zagidull/Documents/fimm_files/publication_data/NCI_Almanac/ComboDrugGrowth_Nov2017_imputed.csv', sep=',',
                   dtype=dtypes) #, parse_dates=parse_dates)
file1.drop(['Unnamed: 0'], inplace=True, axis=1)
file1.iloc[-1,-1] = 'SF-539'  # very last element is 'SF-539\x1a' for some reason


# also need to get na's removed from file1
file1.fillna(pd.Series(-9999, index=file1.select_dtypes(exclude='category').columns), inplace=True)

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

gp = []  # good plates
bp = []  # bad plates
for i,v in file1_grouped_plate:
    combos = v.loc[v.NSC2 != -9999][['NSC1', 'NSC2', 'CELLNAME']].drop_duplicates()  # combos
    singles = v.loc[v.NSC2 == -9999][['NSC1', 'NSC2', 'CELLNAME']].drop_duplicates()  # singles
    combos_cell = set()
    singles_cell = set()
    combos.apply(lambda x: combos_cell.add((x[0], x[2])), axis=1)  # adding NSC1-cell line, iterating over rows
    combos.apply(lambda x: combos_cell.add((x[1], x[2])), axis=1)  # adding NSC2-cell line, iterate over rows
    singles.apply(lambda x: singles_cell.add((x[0], x[2])), axis=1)  # adding NSC1 from single drugs, iter over rows
    temp = combos_cell - singles_cell  # difference should be zero if all single drugs are screened in the same plate as combos
    if not temp:
        gp.append(i)

    else:
        bp.append(i)

#%%
# omg, it s so much easier to just pre-calculate the values for all possible combinations and then do the look up
# fuck
counter = 0
hh = {}
grouped = file1.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])

#%%
for i,v in grouped:
    print(i)
    if i[1] == -9999:
        print(i)
        hh[i] = np.mean(v['PERCENTGROWTHNOTZ'])


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

def good(x):  # for good plates we simply get means from the plates
    return x.groupby(['NSC1', 'NSC2', 'CELLNAME', 'CONC1', 'CONC2'])['PERCENTGROWTHNOTZ'].transform(np.mean)

def bad(x):  # for bad plates we need to get averages of all the screens for that combination of drugs / cell lines / doses
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
#

#########################################

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