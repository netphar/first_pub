"""
this script generates a descriptive analysis of the drugs and cell lines present in drugcomb database
from the following files drugs.new.xlsx and cell_line.xlsx
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

#sns.set(style="ticks")
sns.set(rc={'figure.figsize': (15.7, 7.27)})


#%% first we need to load the files. Let's compare old drugs file wiht with the new drugs file
drugs_new = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drug.new.xlsx')
drugs_old = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drug-2.xlsx')
tissue = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/cell_line.xlsx')

#%% check out differences
drugs_new_id = drugs_new['id']
drugs_old_id = drugs_old['id']
np.setdiff1d(drugs_new_id, drugs_old_id)  # or
diffs = sum(pd.DataFrame(drugs_new_id == drugs_old_id)['id'])  # diffs is equal to the number of drugs. So there are no differences

#%% get counts of categorical data
counts = drugs_new['class.combined'].value_counts()
counts_tissue = tissue['tissue_name'].value_counts()

#%% THIS IS A WORKING ONE
fig1, (ax1, ax2) = plt.subplots(ncols=2)
sns.set_style("whitegrid")
sns.set_context('paper')

# drug types
x = np.char.array(counts.axes[0].to_list())
sizes = counts.values
percent = 100.*sizes/sizes.sum()

labels = ['{0} - {1:1.2f}%'.format(i, j) for i, j in zip(x, percent)]
# only "explode" the 1st slice (i.e. 'Unknowns')
explode = (0.03, 0, 0, 0, 0, 0, 0, 0, 0)
patches, _ = ax1.pie(sizes, startangle=90, radius=1.2, explode=explode, colors=sns.color_palette('deep'),
                     wedgeprops={"edgecolor": "w", 'linewidth': 0, 'linestyle': 'solid', 'antialiased': True})
patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
ax1.legend(patches, labels, loc='center', fontsize=10)
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax1.add_patch(centre_circle)
ax1.set_title('Drug types. Total count = 2285', fontsize=18)

# cell types
x1 = np.char.array(counts_tissue.axes[0].to_list())
sizes1 = counts_tissue.values
percent1 = 100.*sizes1/sizes1.sum()
percent1 = percent1.round(2)
labels1 = ['{0}%'.format(i) for i in percent1]

# data1 = pd.DataFrame(data={'Cell line count': sizes1, 'Tissue type': x1})
#sns.barplot(x='Cell line count', y='Tissue type', data=data1, palette=sns.color_palette('Blues_r', n_colors=len(sizes1)))

ccc = sns.light_palette((60, 75, 60), input="husl", n_colors=len(x1), reverse=True)
ax2.barh(x1, sizes1,   color=sns.color_palette(ccc), align='center', linewidth=0)
ax2.invert_yaxis()
ax2.yaxis.grid(False)

def normaliseCounts(widths,maxwidth):
    '''
    :param widths: this a list of current widths for rectangular object returned by plt.bar / sns.barplot
    :param maxwidth: this is to value to which we are scaling
    :return: list of the same length as input, but scaled according to maxwiddth
    '''
    widths = np.array(widths)/float(maxwidth)
    return widths

columncounts = [70, 70, 70, 70, 70, 70, 70, 70, 70, 70]
widthbars = normaliseCounts(columncounts,100)


for bar, newwidth, value in zip(ax2.patches, widthbars, labels1):
    x = bar.get_x()

    height = bar.get_height()
    centre = x+height/2.

    #bar.set_x(centre-newwidth/2.)
    #bar.set_height(newwidth)
    y_loc = bar.get_y()+bar.get_height()/2.
    ax2.annotate(value, xy=(bar.get_x()+1, y_loc), xytext=(0, 0), ha='center', va='center', fontsize=10, color='black',
                 rotation=0, textcoords='offset points')

ax2.tick_params(labelsize=10)

ax2.set_ylabel('')
ax2.set_xlabel('Count')
ax2.set_title('Cell line types. Total count = 93', fontsize=18)

fig1.tight_layout()
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
filename = 'case1_' + now +'.png'

fig1.savefig(filename, dpi = 600)
plt.close(fig1)
#plt.show()

#%% OG pie, ahem donut
# labels = counts.axes[0].to_list()
# sizes = counts.values
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0)
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, pctdistance=0.7, radius=0.85)
# #draw circle
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# # Equal aspect ratio ensures that pie is drawn as a circle
# ax1.axis('equal')
# plt.tight_layout()
# plt.show()

#%% alternative with legend in the center. Jing's choice
# x = np.char.array(counts.axes[0].to_list())
# sizes = counts.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
# patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
#
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# plt.show()

#%% barplot
# x = np.char.array(counts.axes[0].to_list())
# sizes = counts.values
# data = pd.DataFrame(data={'Drug count': sizes, 'Drug type': x})
# sns.barplot(x='Drug count', y='Drug type', data=data, palette='Spectral')
# plt.show()

# #%% there should be arrows here
# fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
#
# recipe = np.char.array(counts.axes[0].to_list())
#
# data = counts.values
#
# wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
#
# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
# kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
#           bbox=bbox_props, zorder=0, va="center")
#
# for i, p in enumerate(wedges):
#     ang = (p.theta2 - p.theta1)/2. + p.theta1
#     y = np.sin(np.deg2rad(ang))
#     x = np.cos(np.deg2rad(ang))
#     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
#     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
#     kw["arrowprops"].update({"connectionstyle": connectionstyle})
#     ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
#                  horizontalalignment=horizontalalignment, **kw)
#
#
# plt.show()

# #%% forth should have been a pretty polar plot
# # Compute pie slices
# N = len(np.char.array(counts.axes[0].to_list()))
# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = counts.values
# width = np.pi / 4 * np.random.rand(N)
#
# ax = plt.subplot(111, projection='aitoff')
# bars = ax.bar(theta, radii, width=width, bottom=0.0)
#
# # Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor(plt.cm.viridis(r / 10.))
#     bar.set_alpha(0.5)
#
# plt.show()

#%%
# x = np.char.array(counts_tissue.axes[0].to_list())
# sizes = counts_tissue.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
# patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
#
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# plt.show()
#
# #%% barplot
# x = np.char.array(counts_tissue.axes[0].to_list())
# sizes = counts_tissue.values
# data = pd.DataFrame(data={'Cell line count': sizes, 'Tissue type': x})
# sns.barplot(x='Cell line count', y='Tissue type', data=data, palette='pastel')
# plt.show()
#
# #%%
# fig1, (ax1, ax2) = plt.subplots(ncols=2)
# x = np.char.array(counts.axes[0].to_list())
# sizes = counts.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, _ = ax1.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
# patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.25, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# ax1.add_patch(centre_circle)
# ax1.set_title('Drug types', y=1.05, fontsize=10)
#
# x = np.char.array(counts_tissue.axes[0].to_list())
# sizes = counts_tissue.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, _ = ax2.pie(sizes, startangle=90, radius=1.2, explode=explode, colors=current_palette)
# patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.75, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# ax2.add_patch(centre_circle)
#
# ax2.set_title('Cell line types', y=1.05, fontsize=10)
#
# fig1 = plt.gcf()
# fig1.gca().add_artist(centre_circle)
# plt.show()
#
# #%%
# """
# this script generates a descriptive analysis of the drugs present in drugcomb database from the following file
# drugs.new.xlsx
# """
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# sns.set(style="ticks")
# sns.set(rc={'figure.figsize': (15.7, 7.27)})
#
#
# #%% first we need to load the files. Let's compare old drugs file wiht with the new drugs file
# drugs_new = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drug.new.xlsx')
# drugs_old = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/drug-2.xlsx')
# tissue = pd.read_excel('/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/cell_line.xlsx')
#
# #%% check out differences
# drugs_new_id = drugs_new['id']
# drugs_old_id = drugs_old['id']
# np.setdiff1d(drugs_new_id, drugs_old_id)  # or
# diffs = sum(pd.DataFrame(drugs_new_id == drugs_old_id)['id'])  # diffs is equal to the number of drugs. So there are no differences
#
# #%% get counts of categorical data
# counts = drugs_new['class.combined'].value_counts()
# counts_tissue = tissue['tissue_name'].value_counts()
#
# #%% OG pie, ahem donut
# # labels = counts.axes[0].to_list()
# # sizes = counts.values
# # # only "explode" the 2nd slice (i.e. 'Hogs')
# # explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0)
# # fig1, ax1 = plt.subplots()
# # ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, pctdistance=0.7, radius=0.85)
# # #draw circle
# # centre_circle = plt.Circle((0,0),0.70,fc='white')
# # fig = plt.gcf()
# # fig.gca().add_artist(centre_circle)
# # # Equal aspect ratio ensures that pie is drawn as a circle
# # ax1.axis('equal')
# # plt.tight_layout()
# # plt.show()
#
# #%% alternative with legend in the center. Jing's choice
# x = np.char.array(counts.axes[0].to_list())
# sizes = counts.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
# patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
#
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# plt.show()

#%% barplot
# x = np.char.array(counts.axes[0].to_list())
# sizes = counts.values
# data = pd.DataFrame(data={'Drug count': sizes, 'Drug type': x})
# sns.barplot(x='Drug count', y='Drug type', data=data, palette='Spectral')
# plt.show()

# #%% there should be arrows here
# fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
#
# recipe = np.char.array(counts.axes[0].to_list())
#
# data = counts.values
#
# wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)
#
# bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
# kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
#           bbox=bbox_props, zorder=0, va="center")
#
# for i, p in enumerate(wedges):
#     ang = (p.theta2 - p.theta1)/2. + p.theta1
#     y = np.sin(np.deg2rad(ang))
#     x = np.cos(np.deg2rad(ang))
#     horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
#     connectionstyle = "angle,angleA=0,angleB={}".format(ang)
#     kw["arrowprops"].update({"connectionstyle": connectionstyle})
#     ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
#                  horizontalalignment=horizontalalignment, **kw)
#
#
# plt.show()

# #%% forth should have been a pretty polar plot
# # Compute pie slices
# N = len(np.char.array(counts.axes[0].to_list()))
# theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
# radii = counts.values
# width = np.pi / 4 * np.random.rand(N)
#
# ax = plt.subplot(111, projection='aitoff')
# bars = ax.bar(theta, radii, width=width, bottom=0.0)
#
# # Use custom colors and opacity
# for r, bar in zip(radii, bars):
#     bar.set_facecolor(plt.cm.viridis(r / 10.))
#     bar.set_alpha(0.5)
#
# plt.show()

#%%
# x = np.char.array(counts_tissue.axes[0].to_list())
# sizes = counts_tissue.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
# patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
#
# fig = plt.gcf()
# fig.gca().add_artist(centre_circle)
# plt.show()

#%% barplot
# x = np.char.array(counts_tissue.axes[0].to_list())
# sizes = counts_tissue.values
# data = pd.DataFrame(data={'Cell line count': sizes, 'Tissue type': x})
# sns.barplot(x='Cell line count', y='Tissue type', data=data, palette='pastel')
# plt.show()

# #%%
# fig1, (ax1, ax2) = plt.subplots(ncols=2)
# x = np.char.array(counts.axes[0].to_list())
# sizes = counts.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, _ = ax1.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
# patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.25, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.70,fc='white')
# ax1.add_patch(centre_circle)
# ax1.set_title('Drug types', y=1.05, fontsize=10)
#
# x = np.char.array(counts_tissue.axes[0].to_list())
# sizes = counts_tissue.values
# percent = 100.*sizes/sizes.sum()
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# # only "explode" the 2nd slice (i.e. 'Hogs')
# explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# patches, _ = ax2.pie(sizes, startangle=90, radius=1.2, explode=explode, colors=current_palette)
# patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
# fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.75, 0.5),
#            fontsize=8)
# centre_circle = plt.Circle((0,0),0.80,fc='white')
# ax2.add_patch(centre_circle)
#
# ax2.set_title('Cell line types', y=1.05, fontsize=10)
#
# fig1 = plt.gcf()
# fig1.gca().add_artist(centre_circle)
# fig1.savefig('pie_pie.png')
# plt.show()


#%%
# import pickle
# with open('/Users/zagidull/Desktop/filter_input.pickle', 'rb') as f:
#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     data2 = pickle.load(f)
#
# with open('/Users/zagidull/PycharmProjects/test_scientific/filter_input.pickle', 'rb') as f:
#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     data1 = pickle.load(f)
#
#
#
# if data2 == data1:
#     print('yay')

#%%
# sns.set_context("paper")
# sns.set_style("darkgrid")
#
# fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(11.69, 8.27))
#
#
# x = np.char.array(counts.axes[0].to_list())
# sizes = counts.values
# percent = 100.*sizes/sizes.sum()
#
# current_palette = sns.color_palette()
# labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(x, percent)]
# explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
#
# patches, _ = ax1.pie(sizes, startangle=90, radius=1.1, explode=explode, colors=current_palette)
# patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
#                                      key=lambda x: x[2],
#                                      reverse=True))
#
# plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=10)
# centre_circle = plt.Circle((0, 0), 0.70, fc='white')
# ax1.add_patch(centre_circle)
# ax1.set_title('Drug types', y=1, fontsize=18)
# ax1.axis('equal')
#
# plt.show()

