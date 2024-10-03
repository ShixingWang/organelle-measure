#!/usr/bin/env python
# coding: utf-8

# %%
import pandas as pd
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import statsmodels
import statsmodels.api as sm
# import pylab
import seaborn as sns
from pathlib import Path
from organelle_measure.data import read_results

# %% Define the lowess and confidence interval function
def lowess_with_confidence_bounds(x, y, eval_x, N=200, conf_interval=0.95, lowess_kw=None):
    """
    Perform Lowess regression and determine a confidence interval by bootstrap resampling
    """
    # Lowess smoothing
    smoothed = sm.nonparametric.lowess(exog=x, endog=y, xvals=eval_x, **lowess_kw)

    # Perform bootstrap resamplings of the data and evaluate the smoothing at a fixed set of points
    smoothed_values = np.empty((N, len(eval_x)))
    for i in range(N):
        sample = np.random.choice(len(x), len(x), replace=True)
        sampled_x = x[sample]
        sampled_y = y[sample]

        smoothed_values[i] = sm.nonparametric.lowess(exog=sampled_x, endog=sampled_y, xvals=eval_x, **lowess_kw)

    # Get the confidence interval
    sorted_values = np.sort(smoothed_values, axis=0)
    bound = int(N * (1 - conf_interval) / 2)
    bottom = sorted_values[bound - 1]
    top = sorted_values[-bound]

    return smoothed, bottom, top


# %% Define the folders and organelles
size_voxel = 0.1083 * 0.1083 * 0.2
folders = ['']
organelles = {
    'Erg6': "Lipid Droplet",
    'Pex3': "Peroxisome",
    'Sec7': "Golgi Apparatus",
    'Tom70':"Mitochondrion"
}
saveto = "plots/rebuttal_error"

# %% Plot the figures for all combinations of folders and organelles
for organelle in organelles.keys():
    df_filtered = pd.read_csv(f"{saveto}/{organelle}_size_number_data.csv")

    y = size_voxel * np.array(df_filtered['cell-size'])
    x = size_voxel * np.array(df_filtered['size-number-ratio'])

    # Compute the 95% confidence interval
    eval_x = np.linspace(0, x[~np.isnan(x)].max(), 100)
    smoothed, bottom, top = lowess_with_confidence_bounds(x, y, eval_x, lowess_kw={"frac": 0.1})

    # Plot the confidence interval and fit
    # plt.xlim(0, 1000)
    # plt.ylim(0, 4)
    plt.grid()
    xy = np.vstack([x, y])
    valid_data = xy[:, ~np.isnan(xy).any(axis=0)]
    if valid_data.shape[1] > 1:
        density = scipy.stats.gaussian_kde(valid_data)(valid_data)
        light_blue = '#87CEFA'
        black = '#000000'
        cmap_colors = [light_blue, black]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)
        norm = matplotlib.colors.Normalize(vmin=density.min(), vmax=density.max())
        plt.scatter(valid_data[0], valid_data[1], c=density, cmap=cmap, edgecolors=None, s=20, label='Data')
    else:
        plt.scatter(x, y, color='b', edgecolors=None, s=20, label='Data')
    plt.colorbar().set_label('Density')
    plt.plot(eval_x, smoothed, c="k", label='LOWESS ( frac = 0.1 )')
    plt.fill_between(eval_x, bottom, top, alpha=0.5, color="r", label='95% Confidence Interval')
    plt.legend(loc='best', prop={'size': 8})
    plt.title('Cell Size vs. Size Number Ratio')
    plt.ylabel("Cell Size ($\mu m^3$)")
    plt.xlabel("Size Number Ratio ($\mu m^3$)")
    plt.savefig('{}_ratio-cellsize.png'.format(organelle), dpi=600)
    plt.clf()

# %% Plot the figures, flip the axes and take out the lowess regression
for organelle in organelles.keys():
    df_filtered = pd.read_csv(f"{saveto}/{organelle}_size_number_data.csv")

    y = size_voxel * np.array(df_filtered['size-number-ratio'])
    x = size_voxel * np.array(df_filtered['cell-size'])

    # Plot the confidence interval and fit
    plt.grid()
    xy = np.vstack([x, y])
    valid_data = xy[:, ~np.isnan(xy).any(axis=0)]
    if valid_data.shape[1] > 1:
        density = scipy.stats.gaussian_kde(valid_data)(valid_data)
        light_blue = '#87CEFA'
        black = '#000000'
        cmap_colors = [light_blue, black]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors)
        norm = matplotlib.colors.Normalize(vmin=density.min(), vmax=density.max())
        plt.scatter(valid_data[0], valid_data[1], c=density, cmap=cmap, edgecolors=None, s=20, label='Data')
    else:
        plt.scatter(x, y, color='b', edgecolors=None, s=20, label='Data')
    plt.colorbar().set_label('Density')
    plt.legend(loc='best', prop={'size': 8})
    plt.title('Size Number Ratio vs. Cell Size')
    plt.ylabel("Size Number Ratio ($\mu m^3$)")
    plt.xlabel("Cell Size ($\mu m^3$)")
    plt.savefig('{}_cellsize-ratio-noregression.png'.format(organelle), dpi=600)
    plt.clf()


# %%
px_x,px_y,px_z = 0.41,0.41,0.20

subfolders = ["EYrainbow_glucose"]
df_bycell = read_results(Path("./data"),subfolders,(px_x,px_y,px_z))

# %%
