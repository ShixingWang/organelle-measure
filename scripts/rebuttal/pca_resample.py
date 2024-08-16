# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from organelle_measure.data import read_results

# Idea: reample the volume fractions according to the fitted distribution from 
# MCMC simulation, and see how the centroids of conditions, and principal
# components will shift. This will strengthen our confidence of the error of 
# our conclusion.

# The simulation mutates the total volume of a type of organelle in each 
# individual cell, in units of pixel. 
# But what we feed into the PCA is the volume fraction in that cell, in units of um^3/um^3.

# Another problem is that the final volume of ER and vacuoles are 
# measured with further postprocessed after the segmentation. 
# My excuse would be that this same percentage of error will be passed to
# the downstream.

# %% Global Variables
plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '20'
list_colors = {
    "glucose":     [1,2,3,4,0,5],
    "leucine":     [1,2,3,4,0],
    "cell size":   [1,0],
    "PKA pathway": [0,3,2,1],
    "TOR pathway": [0,4,3,2,1]
}

px_x,px_y,px_z = 0.41,0.41,0.20

organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"
]

experiments = {
    "glucose":     "EYrainbow_glucose_largerBF",
    "leucine":     "EYrainbow_leucine_large",
    "cell size":   "EYrainbowWhi5Up_betaEstrodiol",
    "PKA pathway": "EYrainbow_1nmpp1_1st",
    "TOR pathway": "EYrainbow_rapamycin_1stTry"
}
exp_names = experiments.keys()
exp_names = list(exp_names)
exp_folder = [experiments[i] for i in exp_names]

perturbations = {
    "glucose":     "Glucose Concentration",
    "leucine":     "Leucine Concentration",
    "cell size":   "Beta Estrodial Concentration",
    "PKA pathway": "1-nm-pp1 Concentration",
    "TOR pathway": "Rapamycin Concentration"
}

converts = {
    "glucose":     lambda x: f"{x*2/100.:.2f}",
    "leucine":     lambda x: f"{int(x)}",
    "cell size":   lambda x: f"{int(x)}",
    "PKA pathway": lambda x: f"{x/1000.:.1f}",
    "TOR pathway": lambda x: f"{int(x)}",
}

units = {
    "glucose":     "% m/v",
    "leucine":     "mg/L",
    "cell size":   "nM",
    "PKA pathway": "$\\mu$M",
    "TOR pathway": "mg/mL"
}

subfolders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbow_leucine_large",
    "EYrainbow_leucine",
    "EYrainbowWhi5Up_betaEstrodiol"
]

# direction in which the growth rate grows.
extremes = {
    "EYrainbow_glucose":                    [0.,    100.],
    "EYrainbow_glucose_largerBF":           [0.,    100.],
    "EYrainbow_leucine_large":              [0.,    100.],
    "EYrainbowWhi5Up_betaEstrodiol":        [0.,    10.],
    "EYrainbow_rapamycin_1stTry":           [1000., 0.],
    "EYrainbow_rapamycin_CheckBistability": [300.,  0.],
    "EYrainbow_1nmpp1_1st":                 [3000., 0.]
}


# %%
csv_mcmc2575 = pd.read_csv("plots/rebuttal_error/mcmcShankar_25-75.csv")
csv_mcmc2575["std/mean"]  = csv_mcmc2575["standard_deviation"]/csv_mcmc2575["average"]
csv_mcmc2575["diff"]      = csv_mcmc2575["segmented"] - csv_mcmc2575["average"]
csv_mcmc2575["diff/mean"] = csv_mcmc2575["diff"]/csv_mcmc2575["average"]
csv_mcmc2575 = csv_mcmc2575[csv_mcmc2575['diff/mean'].lt(200)]
csv_mcmc2575.dropna(inplace=True)

# %%
by_organlle = csv_mcmc2575[["organelle","std/mean","diff/mean"]].groupby("organelle").mean()

# %%
rng = np.random.default_rng()
multiplier = rng.normal(loc=1.0,scale=0.1,size=100)
# %%

df_bycell = read_results(Path("./data"),subfolders,(px_x,px_y,px_z))
# %%
for t in range(1000):
	continue