# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:33:41 2022

@author: krzys
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy
import os

# Zadania wstępne:
# Dla 3 kolumn o numerycznych wartościach przedstaw:
# 1.box plot
# 2.wyznacz Q1, Q3, IQR oraz SD (standard dev.)

path = r'C:\Users\krzys\Desktop\data science\IV semestr\machine_learning\outliers_exercise'
os.chdir(path)

df = pd.read_csv('houses_data.csv')
desc = df.describe()

def make_boxplot(data, inputs):

    num_inputs = len(inputs)

    fig, axs = plt.subplots(1, num_inputs, figsize=(20,10))

    for i, (ax, curve) in enumerate(zip(axs.flat, inputs), 1):
        sns.boxplot(y=data[curve], ax=ax, color='cornflowerblue', showmeans=True,  
                meanprops={"marker":"o",
                           "markerfacecolor":"white", 
                           "markeredgecolor":"black",
                          "markersize":"10"},
               flierprops={'marker':'o', 
                          'markerfacecolor':'darkgreen',
                          'markeredgecolor':'darkgreen'})
        
        ax.set_title(inputs[i-1])
        ax.set_ylabel('')

    plt.subplots_adjust(hspace=0.15, wspace=1.25)
    plt.show()

numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
num_df = df.select_dtypes(include=numerics)

inputs = list(num_df.columns)
make_boxplot(num_df, inputs)

stats = {}
for col in num_df.columns:
    stats[col] = {'quantiles' : num_df[col].quantile([0.1, 0.25, 0.75, 0.9]),
                  'IQR' : float(num_df[col].quantile([0.75])) - float (num_df[col].quantile([0.25])),
                  'std' : round(num_df[col].std(), 2)}

#%%
# Zadanie główne:

#1.Napisz funkcję, które będą usuwały wartości odstające przy wykorzystaniu:
#   a) log transform
#   b) removing 0.1 & 0.9 percentile
#   c) IQR
#   d) z-score (2 i/lub 3 SD)
#   e) modified Z-score

#2.Porównaj wyniki przez:
#   a) policzenie liczby wystąpień wartości odstających,
#   b) wyznaczenie MAE (kod z poprzednich zajęć)

#a) log transform: ?

#b) percentiles
num_df['Rooms'][(df['Rooms'] > stats['Rooms']['quantiles'][0.1]) & (df['Rooms'] < stats['Rooms']['quantiles'][0.9])]

#c) IQR
iqr = {k: v['IQR'] for (k,v) in stats.items()}
num_df['Rooms'][~((df.Rooms < stats['Rooms']['quantiles'][0.25] - 1.5 * stats['Rooms']['IQR']) | (df.Rooms > stats['Rooms']['quantiles'][0.75] + 1.5 * stats['Rooms']['IQR']))]

#d) Z-SCORE
for col in num_df.columns:
    col_zscore = col + '_zscore'
    num_df[col_zscore] = (num_df[col] - num_df[col].mean())/num_df[col].std(ddof=0)

z_score = num_df.apply(scipy.stats.zscore)

#e) MOD Z-SCORE
#outliers = -3.5>= & 3.5<= or abs mod_z = 3.5
import numpy as np

def mod_z(col: pd.DataFrame, thresh: float=3.5) -> pd.DataFrame:
    med_col = col.median()
    med_abs_dev = (np.abs(col - med_col)).median()
    mod_z = 0.6745 * ((col - med_col) / med_abs_dev)
    mod_z = mod_z[np.abs(mod_z) < thresh]
    return np.abs(mod_z)


