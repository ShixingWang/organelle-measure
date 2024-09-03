import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from organelle_measure.data import read_results

# Global Variables
cmap = plt.get_cmap("tab10")
plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '16'

px_x,px_y,px_z = 0.41,0.41,0.20

list_colors = {
	"glucose":  [1,2,3,4,0,5],
	"rebuttal": [1,2,3,4,0,5],
}

organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"
]
names = {
    "peroxisome":   "Peroxisome",
    "vacuole":      "Vacuole",
    "ER":           "Endoplasmic Reticulum",
    "golgi":        "Golgi Apparatus",
    "mitochondria": "Mitochondrion",
    "LD":           "Lipid Droplet"
}

subfolders  = ["EYrainbow_glucose_largerBF","paperRebuttal"]
experiments = {
	"glucose":  "EYrainbow_glucose_largerBF",
	"rebuttal": "paperRebuttal"
}
extremes = {
    "EYrainbow_glucose_largerBF": [0.,    100.],
	"paperRebuttal":              [0.,    100.],
}

df_bycell = read_results(Path("./data"),subfolders,(px_x,px_y,px_z))

df_3hr = df_bycell[df_bycell['folder'].eq("EYrainbow_glucose_largerBF")]
df_3hr.set_index(["condition","field","idx-cell"],inplace=True)
idx_3hr = df_3hr.groupby(["condition","field","idx-cell"]).count().index
df_pca_3hr = pd.DataFrame(index=idx_3hr,columns=organelles)
for orga in organelles:
	df_pca_3hr[orga] = df_3hr.loc[df_3hr["organelle"].eq(orga),"total-fraction"]
	df_pca_3hr[orga] = (df_pca_3hr[orga] - df_pca_3hr[orga].mean())/df_pca_3hr[orga].std()
df_pca_3hr.reset_index(inplace=True)

df_8hr = df_bycell[df_bycell['folder'].eq("paperRebuttal")]
df_8hr.set_index(["condition","field","idx-cell"],inplace=True)
idx_8hr = df_8hr.groupby(["condition","field","idx-cell"]).count().index
df_pca_8hr = pd.DataFrame(index=idx_8hr,columns=organelles)
for orga in organelles:
	df_pca_8hr[orga] = df_8hr.loc[df_8hr["organelle"].eq(orga),"total-fraction"]
	df_pca_8hr[orga] = (df_pca_8hr[orga] - df_pca_8hr[orga].mean())/df_pca_8hr[orga].std()
df_pca_8hr.reset_index(inplace=True)

for orga in organelles:
	for d,condi in enumerate(np.sort(df_bycell["condition"].unique())):
		print(orga,d,condi)

		plt.figure()
		plt.hist(df_pca_3hr.loc[df_pca_3hr["condition"].eq(condi),orga],density=True,histtype="step",label="3 hours")
		plt.hist(df_pca_8hr.loc[df_pca_8hr["condition"].eq(condi),orga],density=True,histtype="step",label="8 hours")
		plt.title(f"{names[orga]}\n{condi*2/100.:.2f}% w/v glucose")
		plt.xlabel("Normalizaed Volume Fraction")
		plt.legend()
		plt.savefig(
			f"plots/REBUTTAL_8HOURS/normed-volume-fraction_hr3-hr8_{orga}_{str(condi).replace('.','-')}.png",
			dpi=600
		)
		plt.close()