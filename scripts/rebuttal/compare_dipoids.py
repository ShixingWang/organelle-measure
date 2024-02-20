import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from organelle_measure.data import read_results

px_x,px_y,px_z = 0.41,0.41,0.20
organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"
]

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
bycell_6color = read_results(Path("./data/results"),subfolders,(px_x,px_y,px_z))

dfs_read = []
for path in Path("data/results/2024-02-06_paperRebuttal3Colors").glob("*.csv"):
	dfs_read.append(pd.read_csv(str(path)))
df_3color = pd.concat(dfs_read,ignore_index=True)
df_3color["volume-um3"] = (px_x*px_y*px_z)*df_3color["area"]
df_3color.loc[df_3color["organelle"].eq("vacuole"),"volume-um3"] = (
	(px_x * px_y * df_3color.loc[df_3color["organelle"].eq("vacuole"),"area"])
	*2
	*np.sqrt(
		px_x * px_y * df_3color.loc[df_3color["organelle"].eq("vacuole"),"area"]
		/np.pi
	)
)
df_3color["cell-volume-um3"] = (
	(px_x * px_y * df_3color["cell-area"])
	*2
	*np.sqrt(
		px_x * px_y * df_3color["cell-area"]
		/np.pi
	)
)

bycell_3color = df_3color[["organelle","cell-idx","cell-volume-um3"]].groupby(["organelle","cell-idx"]).mean()
bycell_3color["mean-um3"]  = df_3color[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).mean()
bycell_3color["total-um3"] = df_3color[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).sum()
bycell_3color["count"]     = df_3color[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).count()
bycell_3color["volume-fraction"] = bycell_3color["total-um3"] / bycell_3color["cell-volume-um3"]


dfs_read = []
for path in Path("data/results/2024-02-16_rebuttal1color").glob("*.csv"):
	dfs_read.append(pd.read_csv(str(path)))
df_1color = pd.concat(dfs_read,ignore_index=True)
df_1color = pd.concat(dfs_read,ignore_index=True)
df_1color["volume-um3"] = (px_x*px_y*px_z)*df_1color["area"]
df_1color.loc[df_1color["organelle"].eq("vacuole"),"volume-um3"] = (
	(px_x * px_y * df_1color.loc[df_1color["organelle"].eq("vacuole"),"area"])
	*2
	*np.sqrt(
		px_x * px_y * df_1color.loc[df_1color["organelle"].eq("vacuole"),"area"]
		/np.pi
	)
)
df_1color["cell-volume-um3"] = (
	(px_x * px_y * df_1color["cell-area"])
	*2
	*np.sqrt(
		px_x * px_y * df_1color["cell-area"]
		/np.pi
	)
)

bycell_1color = df_1color[["organelle","cell-idx","cell-volume-um3"]].groupby(["organelle","cell-idx"]).mean()
bycell_1color["mean-um3"]  = df_1color[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).mean()
bycell_1color["total-um3"] = df_1color[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).sum()
bycell_1color["count"]     = df_1color[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).count()
bycell_1color["volume-fraction"] = bycell_1color["total-um3"] / bycell_1color["cell-volume-um3"]


