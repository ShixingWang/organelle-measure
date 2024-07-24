import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from pathlib import Path
from sklearn.linear_model import LinearRegression

plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '20'
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

index_bycell = pd.MultiIndex.from_tuples(
	[
		(*index,id_camera) 
		for index in pivot_cell_bycell.index.to_list() 
		for id_camera in df_orga.loc[df_orga["organelle"].eq(index[0]),"camera"].unique()
	],
	names=["organelle","field","idx-cell","camera"]
)

pivot_orga_bycell = pd.DataFrame(index=index_bycell)

pivot_orga_bycell["mean"]  = pivot_orga_bycell_mean
pivot_orga_bycell["count"] = pivot_orga_bycell_nums
pivot_orga_bycell["total"] = pivot_orga_bycell_totl

pivot_orga_bycell.reset_index(inplace=True)
pivot_orga_bycell.set_index(["organelle","field","idx-cell"],inplace=True)
pivot_orga_bycell.loc[:,"cell-area"]   = pivot_cell_bycell.loc[:,"area"]
pivot_orga_bycell.loc[:,"cell-volume"] = pivot_cell_bycell.loc[:,"effective-volume"]

df_bycell = pivot_orga_bycell.reset_index()

organelle_names = {
	"PX": "Peroxisome",
    "VO": "Vacuole",
    "ER": "ER",
    "GL": "Golgi Apparatus",
    "MT": "Mitochondrion",
    "LD": "Lipid Droplet"
}
fluorescence_names = {
	"PX": {
		"DAPI": "mTagBFP2-SKL",
		"FITC": "PEX1-2GFP",
	},
	"VO": {
		"CFP" : "VPH1-mTFP1",
		"FITC": "GFP-PHO8",
	},
	"ER": {
		"FITC" : "SEC61-sfGFP",
		"TRITC": "SS-mCherry-HDEL",
	},
	"GL": {
		"TRITC": "CHS5-mCherry",
		"YFP"  : "SEC7-mCitrine",
	},
	"MT": {
		"FITC" : "COX4-GFP",
		"TRITC": "TOM70-tdTomato",
	},
	"LD": {
		"FITC" : "TGL3-GFP",
		"TRITC": "ERG6-mCherry",
	},
}

for organelle in organelles:
	
	dict_totals = {}
	# dict_counts = {}
	# dict_means  = {}
	for camera in df_bycell.loc[df_bycell["organelle"].eq(organelle),"camera"].unique():
		dict_totals[camera] = df_bycell.loc[
							df_bycell["organelle"].eq(organelle) 
						  & df_bycell["camera"].eq(camera),
						  "total"
						]
		# dict_counts[camera] = df_bycell.loc[
		# 					df_bycell["organelle"].eq(organelle) 
		# 				  & df_bycell["camera"].eq(camera),
		# 				  "count"
		# 				]
		# dict_means[camera] = df_bycell.loc[
		# 					df_bycell["organelle"].eq(organelle) 
		# 				  & df_bycell["camera"].eq(camera),
		# 				  "mean"
		# 				]
	
	cameras = list(dict_totals.keys())
	array_totals = np.vstack((dict_totals[cameras[0]],dict_totals[cameras[1]]))
	if organelle=="VO":
		array_totals[array_totals==0] = np.nan
	array_totals = array_totals[:,~np.isnan(array_totals).any(axis=0)]
	corr_coef = np.corrcoef(array_totals)
	averages = np.mean(array_totals,axis=1)
	stddevis = np.std(array_totals,axis=1)
	normalized = (array_totals - averages.reshape((-1,1)))/stddevis.reshape((-1,1))

	# Overlay of histogram
	plt.rcParams['font.size'] = '20'
	fig,ax = plt.subplots(constrained_layout=True)
	plt.title(f"{organelle_names[organelle]} Total Size")
	ax.hist(
		normalized[0], 20, 
		density=True, histtype='step',
		label=f"{fluorescence_names[organelle][cameras[0]]}"
	)
	ax.hist(
		normalized[1], 20, 
		density=True, histtype='step',
		label=f"{fluorescence_names[organelle][cameras[1]]}"
	)
	ax.legend(fontsize=16)
	ax.set_xlabel("Normalized Organelle Volume")
	ax.set_ylabel("Density")

	positions = [0.68,0.45,0.28,0.22] 
	fsize = 12
	if organelle == "ER":
		positions = [0.78,0.46,0.2,0.15]
		fsize = 10 
	if organelle == "MT":
		positions = [0.71,0.45,0.26,0.21]
		fsize = 11 
	inlet = fig.add_axes(positions)
	inlet.scatter(
		array_totals[0],array_totals[1],
		s=10, facecolor='white',edgecolor=sns.color_palette('tab10')[0],
	)
	inlet.set_title(
		f"$p_{{correlation}}$ = {corr_coef[0,1]:.3f}",
		fontsize=fsize
	)
	inlet.set_xlabel(
		f"{fluorescence_names[organelle][cameras[0]]} ($\\mu m^3$)",
		fontsize=fsize
	)
	inlet.set_ylabel(
		f"{fluorescence_names[organelle][cameras[1]]} ($\\mu m^3$)",
		fontsize=fsize
	)
	# inlet.set_xticks(inlet.get_xticks())
	inlet.set_xticklabels(inlet.get_xticks(), fontsize=fsize)
	# inlet.set_yticks(inlet.get_xticks())
	inlet.set_yticklabels(inlet.get_yticks(), fontsize=fsize)

	plt.savefig(
		f"plots/2labels1organelle/combined_totalsize_{organelle}.png",
		dpi=600
	)


	# # KDE plot of raw data
	# xmin = array_totals[0].min()
	# xmax = array_totals[0].max()
	# ymin = array_totals[1].min()
	# ymax = array_totals[1].max()
	# X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
	# positions = np.vstack([X.ravel(), Y.ravel()])
	# kernel = scipy.stats.gaussian_kde(array_totals)
	# Z = np.reshape(kernel(positions).T, X.shape)

	# plt.figure()
	# plt.title(f"{organelle_names[organelle]} Total Size\ncorrelation coefficient = {corr_coef[0,1]:.3f}")
	# plt.imshow(
	# 	np.rot90(Z), cmap=plt.cm.gist_earth_r,
    #     extent=[xmin, xmax, ymin, ymax], aspect="auto"
	# )
	# plt.scatter(
	# 	array_totals[0],array_totals[1],
	# 	s=2, c='k', marker='.',
	# )
	# plt.xlabel(f"{cameras[0]} ($\\mu m^3$)")
	# plt.ylabel(f"{cameras[1]} ($\\mu m^3$)")
	# plt.savefig(
	# 	f"plots/2labels1organelle/kde_totalsize_{organelle}.png",
	# 	dpi=600
	# )

	# # Scatter plot
	# plt.figure()
	# plt.title(f"{organelle_names[organelle]} Total Size\ncorrelation coefficient = {corr_coef[0,1]:.3f}")
	# plt.scatter(
	# 	array_totals[0],array_totals[1],
	# 	facecolor='white',edgecolor=sns.color_palette('tab10')[0],
	# )
	# plt.xlabel(f"{cameras[0]} ($\\mu m^3$)")
	# plt.ylabel(f"{cameras[1]} ($\\mu m^3$)")
	# plt.savefig(
	# 	f"plots/2labels1organelle/scatter_totalsize_{organelle}.png",
	# 	dpi=600
	# )

	# # Scatter of counter number
	# cameras = list(dict_counts.keys())
	# array_counts = np.vstack((dict_counts[cameras[0]],dict_counts[cameras[1]]))
	# plt.figure()
	# plt.title(f"{organelle_names[organelle]} Counts")
	# plt.scatter(array_counts[0],array_counts[1],facecolor='white', edgecolor=sns.color_palette('tab10')[0])
	# plt.plot(
	# 	[array_counts[0][~np.isnan(array_counts[0])].min(),array_counts[0][~np.isnan(array_counts[0])].max()],
	# 	[array_counts[0][~np.isnan(array_counts[0])].min(),array_counts[0][~np.isnan(array_counts[0])].max()],
	# 	'k--'
	# )
	# plt.xlabel(f"{cameras[0]}")
	# plt.ylabel(f"{cameras[1]}")
	# plt.savefig(
	# 	f"plots/2labels1organelle/counts_{organelle}.png",
	# 	dpi=600
	# )

	# # Scatter of average size
	# cameras = list(dict_means.keys())
	# array_means  = np.vstack((dict_means[cameras[0]],dict_means[cameras[1]]))
	# plt.figure()
	# plt.title(f"{organelle_names[organelle]} Average Size")
	# plt.scatter(array_means[0],array_means[1],facecolor='white', edgecolor=sns.color_palette('tab10')[0])
	# plt.plot(
	# 	[array_means[0][~np.isnan(array_means[0])].min(),array_means[0][~np.isnan(array_means[0])].max()],
	# 	[array_means[0][~np.isnan(array_means[0])].min(),array_means[0][~np.isnan(array_means[0])].max()],
	# 	'k--'
	# )
	# plt.xlabel(f"{cameras[0]} ($\\mu m^3$)")
	# plt.ylabel(f"{cameras[1]} ($\\mu m^3$)")
	# plt.savefig(
	# 	f"plots/2labels1organelle/meansize_{organelle}.png",
	# 	dpi=600
	# )


