import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from organelle_measure.data import read_results

plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '26'
px_x,px_y,px_z = 0.10833, 0.10833, 0.20

organelles = [
    "PX",
    "VO",
    "ER",
    "GL",
    "MT",
    "LD"
]


df_cell = pd.concat(
	(
		pd.read_csv(fcell) 
		for fcell in Path("data/2024-06-25_2colorDiploidMeasure").glob("cell*.csv")
	)
)
df_orga = pd.concat(
	(
		pd.read_csv(forga) 
		for forga in Path("data/2024-06-25_2colorDiploidMeasure").glob("[!c]*.csv")
	)
)

df_cell.loc[:,"effective-volume"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*df_cell.loc[:,"area"]*np.sqrt(df_cell.loc[:,"area"])/np.sqrt(np.pi) 
pivot_cell_bycell = df_cell.set_index(["organelle","field","idx-cell"])


df_orga["volume-micron"] = np.empty_like(df_orga.index)
df_orga.loc[df_orga["organelle"].eq("VO"),"volume-micron"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*df_orga.loc[df_orga["organelle"].eq("VO"),"volume-pixel"]*np.sqrt(df_orga.loc[df_orga["organelle"].eq("VO"),"volume-pixel"])/np.sqrt(np.pi) 
df_orga.loc[df_orga["organelle"].ne("VO"),"volume-micron"] = px_x*px_y*px_z*df_orga.loc[df_orga["organelle"].ne("VO"),"volume-pixel"]

pivot_orga_bycell_mean = df_orga.loc[:,["organelle","field","idx-cell","camera","volume-micron"]].groupby(["organelle","field","idx-cell","camera"]).mean()["volume-micron"]
pivot_orga_bycell_nums = df_orga.loc[:,["organelle","field","idx-cell","camera","volume-micron"]].groupby(["organelle","field","idx-cell","camera"]).count()["volume-micron"]
pivot_orga_bycell_totl = df_orga.loc[:,["organelle","field","idx-cell","camera","volume-micron"]].groupby(["organelle","field","idx-cell","camera"]).sum()["volume-micron"]

pivot_orga_bycell = pd.DataFrame(index=pivot_orga_bycell_mean.index)
pivot_orga_bycell["mean"]  = pivot_orga_bycell_mean
pivot_orga_bycell["count"] = pivot_orga_bycell_nums
pivot_orga_bycell["total"] = pivot_orga_bycell_totl

pivot_orga_bycell.reset_index(inplace=True)
pivot_orga_bycell.set_index(["organelle","field","idx-cell"],inplace=True)
pivot_orga_bycell.loc[:,"cell-area"]   = pivot_cell_bycell.loc[:,"area"]
pivot_orga_bycell.loc[:,"cell-volume"] = pivot_cell_bycell.loc[:,"effective-volume"]

df_bycell = pivot_orga_bycell.reset_index()