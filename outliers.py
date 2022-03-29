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

path = (
    r"C:\Users\krzys\Desktop\data science\IV semestr\machine_learning\outliers_exercise"
)
os.chdir(path)

df = pd.read_csv("houses_data.csv")
desc = df.describe()


def make_boxplot(data, inputs):

    num_inputs = len(inputs)

    fig, axs = plt.subplots(1, num_inputs, figsize=(20, 10))

    for i, (ax, curve) in enumerate(zip(axs.flat, inputs), 1):
        sns.boxplot(
            y=data[curve],
            ax=ax,
            color="cornflowerblue",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "10",
            },
            flierprops={
                "marker": "o",
                "markerfacecolor": "darkgreen",
                "markeredgecolor": "darkgreen",
            },
        )

        ax.set_title(inputs[i - 1])
        ax.set_ylabel("")
    plt.subplots_adjust(hspace=0.15, wspace=1.25)
    plt.show()


numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
num_df = df.select_dtypes(include=numerics)

inputs = list(num_df.columns)
make_boxplot(num_df, inputs)

stats = {}
for col in num_df.columns:
    stats[col] = {
        "quantiles": num_df[col].quantile([0.25, 0.75]),
        "IQR": float(num_df[col].quantile([0.75]))
        - float(num_df[col].quantile([0.25])),
        "std": round(num_df[col].std(), 2),
    }
#%%
# Zadanie główne:

# 1.Napisz funkcję, które będą usuwały wartości odstające przy wykorzystaniu:
#   a) log transform
#   b) removing 0.1 & 0.9 percentile
#   c) IQR
#   d) z-score (2 i/lub 3 SD)
#   e) modified Z-score

# 2.Porównaj wyniki przez:
#   a) policzenie liczby wystąpień wartości odstających,
#   b) wyznaczenie MAE (kod z poprzednich zajęć)

# a) log transform: ?

# b) percentiles
def outliers_percentiles(data: pd.Series, f_quant: float, s_quant: float) -> pd.Series:
    """Function returns pd.Series with outliers and data without outliers
    based on percentiles.
    """

    global outliers_perc
    if type(data) != pd.Series:
        raise TypeError("data must be pd.Series")
    elif type(f_quant) != float or type(s_quant) != float:
        raise TypeError("First quantile and second quantile must be float")
    elif f_quant >= s_quant:
        raise ValueError("First quantile cannot be equal/greater than second")
    else:
        outliers_perc = data[
            ~(
                (data > data.quantile([f_quant])[f_quant])
                & (data < data.quantile([s_quant])[s_quant])
            )
        ]
        return data[
            (data > data.quantile([f_quant])[f_quant])
            & (data < data.quantile([s_quant])[s_quant])
        ]


# c) IQR
def outliers_iqr(data: pd.Series) -> pd.Series:
    """Function returns pd.Series with outliers and data without outliers
    based on IQR.
    """

    global outl_iqr
    if type(data) != pd.Series:
        raise TypeError("data must be pd.Series")
    else:
        Q1 = data.quantile([0.25])[0.25]
        Q3 = data.quantile([0.75])[0.75]
        IQR = Q3 - Q1
        outl_iqr = data[~((data > Q1 - 1.5 * IQR) & (data < Q3 + 1.5 * IQR))]
        data = data[((data > Q1 - 1.5 * IQR) & (data < Q3 + 1.5 * IQR))]
        return data


# d) Z-SCORE
def outliers_zscore(data: pd.Series, thresh: float = 2) -> pd.Series:
    """Function returns pd.Series with outliers and data without outliers
    based on Z-Score.
    """

    global outl_zscore
    if type(data) != pd.Series:
        raise TypeError("data must be pd.Series")
    else:
        z_score = abs(scipy.stats.zscore(data))
        outl_zscore = data[z_score >= thresh]
        return data[z_score < thresh]


# e) MOD Z-SCORE
# outliers = -3.5>= & 3.5<= or abs mod_z = 3.5
import numpy as np


def otuliers_mod_zscore(data: pd.Series, thresh: float = 3.5) -> pd.Series:
    """Function returns pd.Series with outliers and data without outliers
    based on modified Z-Score.
    """

    global outl_mod_zscore
    if type(data) != pd.Series:
        raise TypeError("data must be pd.Series")
    else:
        med_col = data.median()
        med_abs_dev = data.mad()
        mod_z = 0.6745 * ((data - med_col) / med_abs_dev)
        mod_z = mod_z[np.abs(mod_z) < thresh]
        outl_mod_zscore = data[data >= thresh]
        return data[data < thresh]


# porównanie metryk
indexes = ["out_perc", "out_iqr", "out_zscore", "out_mzscore"]
summary = pd.DataFrame(index=indexes)

for col in num_df.columns:
    summary[col] = int()
for index in range(0, len(indexes)):
    for col in num_df.columns:
        if index == 0:
            outliers_percentiles(num_df[col], 0.1, 0.9)
            summary[col].loc[indexes[index]] = len(outliers_perc)
        elif index == 1:
            outliers_iqr(num_df[col])
            summary[col].loc[indexes[index]] = len(outl_iqr)
        elif index == 2:
            outliers_zscore(num_df[col])
            summary[col].loc[indexes[index]] = len(outl_zscore)
        elif index == 3:
            otuliers_mod_zscore(num_df[col])
            summary[col].loc[indexes[index]] = len(outl_mod_zscore)
