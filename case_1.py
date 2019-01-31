"""
this script generates a descriptive analysis of the drugs present in drugcomb database from the following file
drugs.new.xlsx
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="ticks")
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
x = np.char.array(counts.axes[0].to_list())
sizes = counts.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

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
x = np.char.array(counts_tissue.axes[0].to_list())
sizes = counts_tissue.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

#%% barplot
x = np.char.array(counts_tissue.axes[0].to_list())
sizes = counts_tissue.values
data = pd.DataFrame(data={'Cell line count': sizes, 'Tissue type': x})
sns.barplot(x='Cell line count', y='Tissue type', data=data, palette='pastel')
plt.show()

#%%
fig1, (ax1, ax2) = plt.subplots(ncols=2)
x = np.char.array(counts.axes[0].to_list())
sizes = counts.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
patches, _ = ax1.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.25, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')
ax1.add_patch(centre_circle)
ax1.set_title('Drug types', y=1.05, fontsize=10)

x = np.char.array(counts_tissue.axes[0].to_list())
sizes = counts_tissue.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
patches, _ = ax2.pie(sizes, startangle=90, radius=1.2, explode=explode, colors=current_palette)
patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.75, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')
ax2.add_patch(centre_circle)

ax2.set_title('Cell line types', y=1.05, fontsize=10)

fig1 = plt.gcf()
fig1.gca().add_artist(centre_circle)
plt.show()

#%%
"""
this script generates a descriptive analysis of the drugs present in drugcomb database from the following file
drugs.new.xlsx
"""
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sns.set(style="ticks")
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
x = np.char.array(counts.axes[0].to_list())
sizes = counts.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

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
x = np.char.array(counts_tissue.axes[0].to_list())
sizes = counts_tissue.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
patches, texts = plt.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
patches, labels, dummy = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
plt.legend(patches, labels, loc='center', bbox_to_anchor=(0.5, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.show()

#%% barplot
x = np.char.array(counts_tissue.axes[0].to_list())
sizes = counts_tissue.values
data = pd.DataFrame(data={'Cell line count': sizes, 'Tissue type': x})
sns.barplot(x='Cell line count', y='Tissue type', data=data, palette='pastel')
plt.show()

#%%
fig1, (ax1, ax2) = plt.subplots(ncols=2)
x = np.char.array(counts.axes[0].to_list())
sizes = counts.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
patches, _ = ax1.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.25, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')
ax1.add_patch(centre_circle)
ax1.set_title('Drug types', y=1.05, fontsize=10)

x = np.char.array(counts_tissue.axes[0].to_list())
sizes = counts_tissue.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
patches, _ = ax2.pie(sizes, startangle=90, radius=1.2, explode=explode, colors=current_palette)
patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.75, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')
ax2.add_patch(centre_circle)

ax2.set_title('Cell line types', y=1.05, fontsize=10)

fig1 = plt.gcf()
fig1.gca().add_artist(centre_circle)
fig1.savefig('pie_pie.png')
plt.show()

#%%
fig1, (ax1, ax2) = plt.subplots(ncols=2)
x = np.char.array(counts.axes[0].to_list())
sizes = counts.values
percent = 100.*sizes/sizes.sum()
current_palette = sns.color_palette()
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.1, 0, 0, 0, 0, 0, 0, 0, 0)
patches, _ = ax1.pie(sizes, startangle=90, radius=1.2, explode=explode,colors=current_palette)
patches, labels, _ = zip(*sorted(zip(patches, labels, sizes),
                                     key=lambda x: x[2],
                                     reverse=True))
fig1.legend(patches, labels, loc='center', bbox_to_anchor=(0.25, 0.5),
           fontsize=8)
centre_circle = plt.Circle((0,0),0.70,fc='white')
ax1.add_patch(centre_circle)
ax1.set_title('Drug types', y=1.05, fontsize=10)



x = np.char.array(counts_tissue.axes[0].to_list())
sizes = counts_tissue.values
data = pd.DataFrame(data={'Cell line count': sizes, 'Tissue type': x})
sns.barplot(x='Cell line count', y='Tissue type', data=data, palette='pastel')
plt.show()

fig = ax1.get_figure()
fig.savefig('pie_bar.png')