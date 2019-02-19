#%%
import time


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils, plot_model
from keras import backend as K


from distutils.version import LooseVersion as LV
from keras import __version__

from keras.utils.vis_utils import model_to_dot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

print('Using Keras version:', __version__, 'backend:', K.backend())
assert(LV(__version__) >= LV("2.0.0"))

#%%
if K.backend() == "tensorflow":
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name == '':
        device_name = "None"
    # config = tf.ConfigProto()
    # config.intra_op_parallelism_threads = 4
    # config.inter_op_parallelism_threads = 4
    # tf.Session(config=config)
#    config = tf.ConfigProto(device_count={"CPU": 3})
#    K.tensorflow_backend.set_session(tf.Session(config=config))
    print('Using TensorFlow version:', tf.__version__, ', GPU:', device_name)

#%%
import os
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

# %%
import os
# this retarded way is something we are doing to match filenames to cell lines
categ = '/Users/zagidull/Documents/fimm_files/chemical_similarity/classifier/'
directory = os.fsencode(categ+'averages_17012019')

count = 0
urgh = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".csv"):
        count = count + 1
        urgh.append(filename)
        #        print(filename)
        continue
    else:
        continue
print(count, urgh)

# %%
import re
urgh_cleaned = {}
for file in urgh:
    s = file
    match = re.findall("_(.*?)_", s)
    urgh_cleaned[s] = match[0]
#    print(match[0])
print(len(urgh_cleaned), urgh_cleaned)


# %%
def sorter(x, name):
    out = x.loc[x['cell_line_name'] == name]
    out = out.loc[out['drug_row'] != 'ARSENIC TRIOXIDE']
    out = out.loc[out['drug_col'] != 'ARSENIC TRIOXIDE']
    return out

# %%
values_synergy_new = pd.read_csv(
    categ+'drugs_synergy_css_09012019_updatedCSS.csv',
    sep=';')
names = np.unique(values_synergy_new['cell_line_name'])

smiles_holder = pd.read_csv(
    categ+'summary_css_dss_ic50_synergy_smiles.csv',
    sep=';')
smiles_holder.drop(columns=['Unnamed: 0'], inplace=True)
all_holder = {name: sorter(smiles_holder, name) for name in names}
#%%
os.chdir(categ +'averages_17012019')

to_drop = ['drug_row', 'drug_col', 'block_id', 'drug_row_id', 'drug_col_id', 'cell_line_id', 'synergy_zip',
           'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'ic50_row', 'ic50_col', 'cell_line_name', 'smiles_row',
           'smiles_col','dss_row','dss_col']

for i, dt in urgh_cleaned.items():
    print(dt)
    if dt == 'KBM-7':
        test = pd.read_csv(i, sep=';')
        print(i, dt)
        v = pd.merge(left=test, right=all_holder[dt], left_on=['Drug1', 'Drug2'], right_on=['drug_row', 'drug_col'],
                     how='inner')
        v.drop(columns=['drug_row', 'drug_col'], inplace=True)
        v.rename(columns={'Drug1': 'drug_row', 'Drug2': 'drug_col'}, inplace=True)
        v.drop(to_drop, axis=1, inplace=True)
        x_test = v.sample(frac=.1)
        x_train = v.drop(x_test.index)
        y_test = x_test.pop('css')
        y_train = x_train.pop('css')

#%% alternative to preprocessing stuff, as seen above
import pickle
with open('/Users/zagidull/Desktop/KBM7.pickle', 'rb') as f:
    x = pickle.load(f)

#%%
x_test = x.sample(frac=.1)
x_train = x.drop(x_test.index)
y_test = x_test.pop('css')
y_train = x_train.pop('css')

# x_test, x_train, y_test, y_train = np.array(x_test), np.array(x_train), np.array(y_test), np.array(y_train)

#%% making MLP's
def make_model(dropout = 0, hidden_unit = None):
    model = Sequential()
    model.add(Dense(1, input_dim=2048))
    model.add(Activation('relu'))

    model.add(Dense(hidden_unit))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mse',
                     optimizer='rmsprop')
    return model

#%%
a = np.linspace(0.01, 0.4, 5)
for i,v in enumerate(a):
    a[i] = round(v,2)
b = np.linspace(50, 5000, 5, dtype=np.int32)
#%%
K.clear_session()

models = {}
for i in a:
    for j in b:
        key = str(i)+" "+str(j)
        models[key] = make_model(i, j)

#%% training
epochs = 5 # one epoch takes about 3 seconds
history = {}
for i,v in models.items():
    print(i)
    history[i] = v.fit(np.array(x_train), np.array(y_train), epochs=epochs, batch_size=1, verbose=2)

#%% fitting and eval'ing linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lr_split = LinearRegression(fit_intercept=False, n_jobs=7)
lr_split.fit(np.array(x_train), np.array(y_train))

lr_rmse = np.sqrt(mean_squared_error(np.array(y_test), lr_split.predict(np.array(x_test))))
print('rmse of linear model is: {:.2f}'.format(lr_rmse))

#%%
plt.figure(figsize=(12,8))
for i,v in history.items():
    legend = str(i)
    plt.plot(v.epoch,np.sqrt(v.history['loss']), label=legend)
plt.legend(loc='upper right')
plt.xlabel('epochs count')
plt.title('rmse of different MLPs')
plt.axhline(lr_rmse, label='linear regression')
plt.show()
#%% eval on test split of MLP
scores = {}
for i,v in models.items():

    scores[i] = np.sqrt(v.evaluate(np.array(x_test),
                              np.array(y_test),
                              verbose=2))
    print("%s: %.2f using %s" % (v.metrics_names[0], scores[i],i))
print('rmse of linear model is: {:.2f}'.format(lr_rmse))

#%%
# getting lowest 5 setups and training them for 15 epochs each
epochs = 15
for a,v in models.items():
    if a in sorted(scores, key=scores.get, reverse=False)[:5]:
        print(a)
        history[a] = v.fit(np.array(x_train), np.array(y_train), epochs=epochs, batch_size=1, verbose=2)
#%% let's get brain one, glioblastoma e.g.  SF 295
os.chdir(categ +'averages_17012019')

to_drop = ['drug_row', 'drug_col', 'block_id', 'drug_row_id', 'drug_col_id', 'cell_line_id', 'synergy_zip',
           'synergy_bliss', 'synergy_loewe', 'synergy_hsa', 'ic50_row', 'ic50_col', 'cell_line_name', 'smiles_row',
           'smiles_col','dss_row','dss_col']

for i, dt in urgh_cleaned.items():
    print(dt)
    if dt == 'SF-295':
        test = pd.read_csv(i, sep=';')
        print(i, dt)
        v1 = pd.merge(left=test, right=all_holder[dt], left_on=['Drug1', 'Drug2'], right_on=['drug_row', 'drug_col'],
                     how='inner')
        v1.drop(columns=['drug_row', 'drug_col'], inplace=True)
        v1.rename(columns={'Drug1': 'drug_row', 'Drug2': 'drug_col'}, inplace=True)
        v1.drop(to_drop, axis=1, inplace=True)
        x1_test = v1.sample(frac=.1)
        x1_train = v1.drop(x1_test.index)
        y1_test = x1_test.pop('css')
        y1_train = x1_train.pop('css')

#%% lin r
lr1_split = LinearRegression(fit_intercept=False, n_jobs=7)
lr1_split.fit(np.array(x1_train), np.array(y1_train))

lr1_rmse = np.sqrt(mean_squared_error(np.array(y1_test), lr1_split.predict(np.array(x1_test))))
print('rmse of linear model is: {:.2f}'.format(lr1_rmse))

#%%
a1 = np.linspace(0.01, 0.2, 3)
for i,v in enumerate(a1):
    a1[i] = round(v,2)
b1 = np.linspace(50, 500, 3, dtype=np.int32)

#%%
from functools import reduce

def factors(n):
    return set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))
#%%
def make_model1(dropout = 0, hidden_unit = None):
    model = Sequential()
    model.add(Dense(1, input_dim=2048))
    model.add(Activation('relu'))

    model.add(Dense(hidden_unit))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mse',
                     optimizer='rmsprop')
    return model
#%%
models1 = {}
for i in a1:
    for j in b1:
        key = str(i)+" "+str(j)
        models1[key] = make_model1(i, j)

#%% training
epochs = 500 # one epoch takes about 3 seconds
history1 = {}
for i,v in models1.items():
    print(i)
    history1[i] = v.fit(np.array(x1_train), np.array(y1_train), epochs=epochs, batch_size=4339, verbose=2)

#%%
plt.figure(figsize=(12,8))
for i,v in history1.items():
    legend = str(i)
    plt.plot(v.epoch,np.sqrt(v.history['loss']), label=legend)
plt.legend(loc='upper right')
plt.xlabel('epochs count')
plt.title('rmse of different MLPs')
plt.axhline(lr1_rmse, label='linear regression')
plt.show()

#%%
#%% eval on test split of MLP
scores1 = {}
for i,v in models1.items():

    scores1[i] = np.sqrt(v.evaluate(np.array(x1_test),
                              np.array(y1_test),
                              verbose=2))
    print("%s: %.2f using %s" % (v.metrics_names[0], scores1[i],i))

#%%
def make_model2(dropout = 0, hidden_unit = None):
    model = Sequential()
    model.add(Dense(1, input_dim=2048))
    model.add(Activation('relu'))

    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(hidden_unit))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))

    model.add(Dense(units=1, activation='linear'))

    model.compile(loss='mse',
                     optimizer='rmsprop')
    return model

#%%
models2 = {}
for i in [0.1]:
    for j in [50,100,150]:
        key = str(i)+" "+str(j)
        models2[key] = make_model2(i, j)

#%% training
epochs = 150 # one epoch takes about 3 seconds
history2 = {}
for i,v in models2.items():
    print(i)
    history2[i] = v.fit(np.array(x1_train), np.array(y1_train), epochs=epochs, batch_size=4339, verbose=2)

#%%
#%%
plt.figure(figsize=(12,8))
for i,v in history2.items():
    legend = str(i)
    plt.plot(v.epoch,np.sqrt(v.history['loss']), label=legend)
plt.legend(loc='upper right')
plt.xlabel('epochs count')
plt.title('rmse of different MLPs')
plt.axhline(lr1_rmse, label='linear regression')
plt.show()

#%%
#%% eval on test split of MLP
scores2 = {}
for i,v in models2.items():

    scores2[i] = np.sqrt(v.evaluate(np.array(x1_test),
                              np.array(y1_test),
                              verbose=2))
    print("%s: %.2f using %s" % (v.metrics_names[0], scores2[i],i))
print('rmse of linear model is: {:.2f}'.format(lr1_rmse))

#%%plotting a sigmoid functin
import matplotlib.pylab as plt
import numpy as np
x = np.arange(-8, 8, 0.1)
f = 1 / (4 + 7*np.exp(-x)+1)
plt.plot(x, f, color='r')
plt.ylim(-0.2, 1.2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

#%%
import matplotlib.pylab as plt
import numpy as np
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f, color = 'b')
plt.ylim(-0.2, 1.2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()