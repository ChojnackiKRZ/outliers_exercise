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

def make_boxplot(welldata, inputs):

    num_inputs = len(inputs)

    fig, axs = plt.subplots(1, num_inputs, figsize=(20,10))

    for i, (ax, curve) in enumerate(zip(axs.flat, inputs), 1):
        sns.boxplot(y=welldata[curve], ax=ax, color='cornflowerblue', showmeans=True,  
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
inputs.pop(0) # remove the well name from the columns list

make_boxplot(num_df, inputs)

quantiles = {}
for col in num_df:
    quantiles[col] = num_df[col].quantile([0.25, 0.75])