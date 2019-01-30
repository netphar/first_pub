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

#%% check out differences
drugs_new_id = drugs_new['id']
drugs_old_id = drugs_old['id']
np.setdiff1d(drugs_new_id, drugs_old_id)  # or
diffs = sum(pd.DataFrame(drugs_new_id == drugs_old_id)['id'])  # diffs is equal to the number of drugs. So there are no differences

#%% get counts of categorical data
counts = drugs_new['class.combined'].value_counts()

#%% OG pie, ahem donut
labels = counts.axes[0].to_list()
sizes = counts.values
# only "explode" the 2nd slice (i.e. 'Hogs')
explode = (0.0, 0, 0, 0, 0, 0, 0, 0, 0)
fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, pctdistance=0.7, radius=0.85)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')
plt.tight_layout()
plt.show()

#%% alternative with legend in the center
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
x = np.char.array(counts.axes[0].to_list())
sizes = counts.values
data = pd.DataFrame(data={'Drug count': sizes, 'Drug type': x})
sns.barplot(x='Drug count', y='Drug type', data=data, palette='Spectral')
plt.show()

#%% there should be arrows here
fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))

recipe = np.char.array(counts.axes[0].to_list())

data = counts.values

wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(xycoords='data', textcoords='data', arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(recipe[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                 horizontalalignment=horizontalalignment, **kw)


plt.show()

#%% forth should have been a pretty polar plot
# Compute pie slices
N = len(np.char.array(counts.axes[0].to_list()))
theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
radii = counts.values
width = np.pi / 4 * np.random.rand(N)

ax = plt.subplot(111, projection='aitoff')
bars = ax.bar(theta, radii, width=width, bottom=0.0)

# Use custom colors and opacity
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.viridis(r / 10.))
    bar.set_alpha(0.5)

plt.show()

#$$
# todo: create the same kind of plot for cells. Jing likes version with legend in the center