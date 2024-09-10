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
organelle_names = {
    "peroxisome"  : "Peroxisome"  ,
    "vacuole"     : "Vacuole"    ,
    "ER"          : "ER"   ,
    "golgi"       : "Golgi Apparatus" ,
    "mitochondria": "Mitochondrion",
    "LD"          : "Lipid Droplet"  ,
}
markers = {
    "peroxisome"  : "mTagBFP2-SKL"  ,
    "vacuole"     : "VPH1-mTFP1"    ,
    "ER"          : "SEC61-sfGFP"   ,
    "golgi"       : "SEC7-mCitrine" ,
    "mitochondria": "TOM70-tdTomato",
    "LD"          : "ERG6-mCherry"  ,
}

# 6-color cells segmented by 6-color ilastik
subfolders = ["EYrainbow_glucose_largerBF"]
bycell_6segmented_by6 = read_results(Path("./data/"),subfolders,(px_x,px_y,px_z))
bycell_6segmented_by6 = bycell_6segmented_by6.loc[
	bycell_6segmented_by6["condition"].eq(100)
#   & bycell_6segmented_by6["field"].eq(6)
]
bycell_6segmented_by6.rename(
	columns={
		"idx-cell"      : "cell-idx",
		"cell-volume"   : "cell-volume-um3",
		"mean"          : "mean-um3",
		"total"         : "total-um3",
		"total-fraction": "volume-fraction"
	},
	inplace=True
)

# # 6-color 8-hour data
# subfolders = ["paperRebuttal"]
# bycell_8hours = read_results(Path("./data/"),subfolders,(px_x,px_y,px_z))
# bycell_8hours = bycell_8hours.loc[bycell_8hours["condition"].eq(100)]
# bycell_8hours.rename(
# 	columns={
# 		"idx-cell"      : "cell-idx",
# 		"cell-volume"   : "cell-volume-um3",
# 		"mean"          : "mean-um3",
# 		"total"         : "total-um3",
# 		"total-fraction": "volume-fraction"
# 	},
# 	inplace=True
# )

# 1-color cells segmented by 1-color ilastik
dfs_read = []
for	path in Path("data/rebuttal_diploid_comparison/1color-cells_1color-10param-ilastik").glob(f"SameParamAs6*.csv"):
	dfs_read.append(pd.read_csv(str(path)))
df_1segmented_by1 = pd.concat(dfs_read,ignore_index=True)
df_1segmented_by1.loc[df_1segmented_by1["organelle"].ne("vacuole"),"field"] = 1
df_1segmented_by1["volume-um3"] = (px_x*px_y*px_z)*df_1segmented_by1["area"]
df_1segmented_by1.loc[df_1segmented_by1["organelle"].eq("vacuole"),"volume-um3"] = (
	(px_x * px_y * df_1segmented_by1.loc[df_1segmented_by1["organelle"].eq("vacuole"),"area"])
	*2
	*np.sqrt(
		px_x * px_y * df_1segmented_by1.loc[df_1segmented_by1["organelle"].eq("vacuole"),"area"]
		/np.pi
	)
)
df_1segmented_by1["cell-volume-um3"] = (
	(px_x * px_y * df_1segmented_by1["cell-area"])
	*2
	*np.sqrt(
		px_x * px_y * df_1segmented_by1["cell-area"]
		/np.pi
	)
)

bycell_1segmented_by1 = df_1segmented_by1[["organelle","field","cell-idx","cell-volume-um3"]].groupby(["organelle","field","cell-idx"]).mean()
bycell_1segmented_by1["mean-um3"]  = df_1segmented_by1[["organelle","field","cell-idx","volume-um3"]].groupby(["organelle","field","cell-idx"]).mean()
bycell_1segmented_by1["total-um3"] = df_1segmented_by1[["organelle","field","cell-idx","volume-um3"]].groupby(["organelle","field","cell-idx"]).sum()
bycell_1segmented_by1["count"]     = df_1segmented_by1[["organelle","field","cell-idx","volume-um3"]].groupby(["organelle","field","cell-idx"]).count()
bycell_1segmented_by1["volume-fraction"] = bycell_1segmented_by1["total-um3"] / bycell_1segmented_by1["cell-volume-um3"]
bycell_1segmented_by1.reset_index(inplace=True)

# for organelle in organelles:
# 	print(organelle, len(bycell_1segmented_by1[bycell_1segmented_by1['organelle'].eq(organelle)]))
# >>> peroxisome 202
# >>> vacuole 403
# >>> ER 518
# >>> golgi 433
# >>> mitochondria 275
# >>> LD 356

# # 6-color cells segmented by 1-color ilastik
# dfs_read = []
# for path in Path("data/rebuttal_diploid_comparison/6color-cells_1color-10param-ilastik").glob(f"*.csv"):
# 	df = pd.read_csv(str(path))
# 	if len(df) > 0:
# 		dfs_read.append(df)
# df_6segmented_by1 = pd.concat(dfs_read,ignore_index=True)
# df_6segmented_by1["cell-idx"] = df_6segmented_by1["field"].astype(str) + "-" + df_6segmented_by1["cell-idx"].astype(str)
# df_6segmented_by1["volume-um3"] = (px_x*px_y*px_z)*df_6segmented_by1["area"]
# df_6segmented_by1.loc[df_6segmented_by1["organelle"].eq("vacuole"),"volume-um3"] = (
# 	(px_x * px_y * df_6segmented_by1.loc[df_6segmented_by1["organelle"].eq("vacuole"),"area"])
# 	*2
# 	*np.sqrt(
# 		px_x *  px_y * df_6segmented_by1.loc[df_6segmented_by1["organelle"].eq("vacuole"),"area"]
# 		/np.pi
# 	)
# )
# df_6segmented_by1["cell-volume-um3"] = (
# 	(px_x * px_y * df_6segmented_by1["cell-area"])
# 	*2
# 	*np.sqrt(
# 		px_x * px_y * df_6segmented_by1["cell-area"]
# 	    /np.pi
# 	)
# )

# bycell_6segmented_by1 = df_6segmented_by1[["organelle","cell-idx","cell-volume-um3"]].groupby(["organelle","cell-idx"]).mean()
# bycell_6segmented_by1["mean-um3"]  = df_6segmented_by1[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).mean()
# bycell_6segmented_by1["total-um3"] = df_6segmented_by1[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).sum()
# bycell_6segmented_by1["count"]     = df_6segmented_by1[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).count()
# bycell_6segmented_by1["volume-fraction"] = bycell_6segmented_by1["total-um3"] / bycell_6segmented_by1["cell-volume-um3"]
# bycell_6segmented_by1.reset_index(inplace=True)


# # 1-color cells segmented by 6-color ilastik
# dfs_read = []
# for path in Path("data/rebuttal_diploid_comparison/1color-cells_6color-10param-ilastik").glob("*.csv"):
# 	df = pd.read_csv(str(path))
# 	if len(df) > 0:
# 		dfs_read.append(df)
# df_1segmented_by6 = pd.concat(dfs_read,ignore_index=True)
# df_1segmented_by6["volume-um3"] = (px_x*px_y*px_z)*df_1segmented_by6["area"]
# df_1segmented_by6.loc[df_1segmented_by6["organelle"].eq("vacuole"),"volume-um3"] = (
# 	(px_x * px_y * df_1segmented_by6.loc[df_1segmented_by6["organelle"].eq("vacuole"),"area"])
# 	*2
# 	*np.sqrt(
# 		px_x * px_y * df_1segmented_by6.loc[df_1segmented_by6["organelle"].eq("vacuole"),"area"]
# 		/np.pi
# 	)
# )
# df_1segmented_by6["cell-volume-um3"] = (
# 	(px_x * px_y * df_1segmented_by6["cell-area"])
# 	*2
# 	*np.sqrt(
# 		px_x * px_y * df_1segmented_by6["cell-area"]
# 	    /np.pi
# 	)
# )

# bycell_1segmented_by6 = df_1segmented_by6[["organelle","cell-idx","cell-volume-um3"]].groupby(["organelle","cell-idx"]).mean()
# bycell_1segmented_by6["mean-um3"]  = df_1segmented_by6[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).mean()
# bycell_1segmented_by6["total-um3"] = df_1segmented_by6[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).sum()
# bycell_1segmented_by6["count"]     = df_1segmented_by6[["organelle","cell-idx","volume-um3"]].groupby(["organelle","cell-idx"]).count()
# bycell_1segmented_by6["volume-fraction"] = bycell_1segmented_by6["total-um3"] / bycell_1segmented_by6["cell-volume-um3"]
# bycell_1segmented_by6.reset_index(inplace=True)


# Plot
plt.rcParams["figure.autolayout"]=True
plt.rcParams['font.size'] = '20'

for organelle in organelles:
	legends = [
		f"{markers[organelle]} only",
		"Rainbow Yeast",
	]
	legend_size = '16'
	if organelle=="mitochondria":
		legend_size = '11'
	for property in [
						# "mean-um3",
						"total-um3",
						# "volume-fraction","count"
					]:
		# # bar plot
		# plt.figure()
		# for d,dataset in enumerate([
		# 	bycell_1segmented_by1,
		# 	bycell_6segmented_by6,
		# 	# bycell_8hours,
		# ]):
		# 	plt.bar(
		# 	    [d], height=[dataset.loc[dataset["organelle"].eq(organelle),property].mean()], 
		# 	           yerr=[dataset.loc[dataset["organelle"].eq(organelle),property].std()],
		# 	)
		# plt.xticks(
		# 	ticks=np.arange(2),
		# 	labels=legends
		# )
		# name = property.replace('um3','volume').replace("count","number").replace('-',' ')
		# y_label = r"Volume ($\mu$m$^3$)" if "-um" in property else name
		# plt.ylabel(y_label)
		# plt.title(organelle)
		# plt.savefig(
		# 	f"plots/rebuttal_diploid_comparison/bar_{organelle}_{name}.png",
		# 	dpi=600
		# )

		# distribution histogram
		plt.figure()
		to_switch_xtick = []
		for d,dataset in enumerate([
			bycell_1segmented_by1,
			bycell_6segmented_by6,
			# bycell_8hours,
		]):
			to_switch_xtick.append(dataset.loc[dataset["organelle"].eq(organelle),property].values)
		to_switch_xtick = np.hstack(to_switch_xtick)
		median = np.median(to_switch_xtick)
		for d,dataset in enumerate([
			bycell_1segmented_by1,
			bycell_6segmented_by6,
			# bycell_8hours,
		]):		
			distrib = dataset.loc[dataset["organelle"].eq(organelle),property]
			# distrib = (distrib - distrib.mean())/distrib.std()
			distrib = distrib / median
			binned  = np.arange(-0.5,distrib.max()) if property=="count" else int(np.sqrt(len(distrib))) 
			plt.hist(
				distrib,
				bins=binned, histtype="step", density=True,
				label=f"{legends[d]}"
			)
		plt.legend(fontsize=legend_size)
		name = property.replace('-um3',' Volume').replace("count","number").replace('-',' ').title()
		# x_label = r"Volume ($\mu$m$^3$)" if "-um" in property else name
		x_label = name
		plt.xlabel(f"Normalized {x_label}")
		plt.ylabel("Density")
		plt.title(f"{organelle_names[organelle]} {name}")
		plt.savefig(
			f"plots/rebuttal_diploid_comparison/distribution-median-xtick_{organelle}_{name}.png",
			dpi=600
		)

