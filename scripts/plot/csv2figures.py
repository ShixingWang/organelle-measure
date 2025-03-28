import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import container
from matplotlib.figure import figaspect
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from organelle_measure.data import read_results


# Global Variables
cmap = plt.get_cmap("tab10")
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


# READ FILES
df_bycell = read_results(Path("./data"),subfolders,(px_x,px_y,px_z))


# DATAFRAME FOR CORRELATION COEFFICIENT

pv_bycell = df_bycell.set_index(['folder','condition','field','idx-cell'])
df_corrcoef = pd.DataFrame(index=pv_bycell.loc[pv_bycell["organelle"].eq("ER")].index)
df_corrcoef.loc[:,'cell length'] = np.sqrt(pv_bycell.loc[pv_bycell["organelle"].eq("ER"),'cell-area']/np.pi)
df_corrcoef.loc[:,'cell area'] = pv_bycell.loc[pv_bycell["organelle"].eq("ER"),'cell-area']
df_corrcoef.loc[:,'cell volume'] = pv_bycell.loc[pv_bycell["organelle"].eq("ER"),'cell-volume']
properties = []
rename_dict = {
    'mean'          : "average volume",
    'count'         : "number",
    'total'         : "total volume",
    'total-fraction': "volume fraction"
}
for orga in [*organelles,"non-organelle"]:
    for prop in ['mean','count','total','total-fraction']:
        if (orga in ["ER","vacuole","non-organelle"]) and (prop in ["count","mean"]):
            continue
        prop_new = f"{orga} {rename_dict[prop]}"
        properties.append(prop_new)
        df_corrcoef.loc[:,prop_new] = pv_bycell.loc[pv_bycell["organelle"].eq(orga),prop]
df_corrcoef.reset_index(inplace=True)

# Kullback–Leibler_divergence of different conditions
df_entropies = []
for exp in exp_names:
    folder = experiments[exp]
    num_bin = 4
    organelles_kl = [f"{orga} volume fraction" for orga in organelles]
    df_kldiverge = df_corrcoef.loc[df_corrcoef["folder"].eq(folder),["condition",*organelles_kl]]
    # get grids in 6 dimensional space
    grids = np.array(list(map(
                lambda x:np.percentile(
                    df_kldiverge[x],
                    q=np.arange(0,100,100/num_bin)
                ),
                organelles_kl
            )))
    probabilities = {}
    probs_dummy = {}
    for condi in np.sort(df_kldiverge["condition"].unique()):
        posits = np.array(list(map(
                    lambda x:np.digitize(
                        df_kldiverge.loc[df_kldiverge["condition"].eq(condi),x[0]],
                        x[1]
                    ),
                    zip(organelles_kl,grids)
                  ))) - 1
        probs = np.zeros(tuple(num_bin for i in range(len(organelles_kl))))
        for count in np.transpose(posits):
            probs[tuple(count)] += 1.
        probs += 10**(-20)
        probs = probs/np.sum(probs)

       
        marginals = np.array([
            np.sum(
                probs,
                axis=tuple(
                    np.delete(np.arange(len(organelles_kl)),i)
                )
            )
            for i in range(len(organelles_kl))
        ])
        
        dummy = np.zeros(tuple(num_bin for i in range(len(organelles_kl))))
        for i0 in range(4):
            for i1 in range(4):
                for i2 in range(4):
                    for i3 in range(4):
                        for i4 in range(4):
                            for i5 in range (4):
                                dummy[(i0,i1,i2,i3,i4,i5)] = marginals[0,i0]*marginals[1,i1]*marginals[2,i2]*marginals[3,i3]*marginals[4,i4]*marginals[5,i5]
        
        probabilities[condi] = probs.flatten()
        probs_dummy[condi] = dummy.flatten()

    normal = extremes[folder][-1]
    divergence = []
    dvgn_dummy = []
    entropy    = []
    entr_dummy = []
    entropy_difference = []
    entropy_diff_dummy = []
    for condi in probabilities.keys():
        divergence.append(
            np.sum(scipy.special.rel_entr(probabilities[condi],probabilities[normal]))
        )
        dvgn_dummy.append(
            np.sum(scipy.special.rel_entr(probs_dummy[condi],probabilities[normal]))
        )
        entropy.append(
            np.sum(
                np.array(list(map(
                    lambda x: 0 if x<10**(-19) else -x*np.log(x),
                    probabilities[condi]
                )))
            )
        )
        entr_dummy.append(
            np.sum(
                np.array(list(map(
                    lambda x: 0 if x<10**(-19) else -x*np.log(x),
                    probs_dummy[condi]
                )))
            )
        )
    df_entropy = pd.DataFrame({
        "experiment": exp,
        "folder":     folder,
        "condition":  np.sort(df_kldiverge["condition"].unique()),
        "entropy":    entropy,
        "KL_divergence": divergence,
        "entropy_dummy": entr_dummy,
        "KL_divergence_dummy": dvgn_dummy
    })
    df_entropy.reset_index(inplace=True)
    df_entropies.append(df_entropy)
    df_entropies[-1]["entropy_difference"] = df_entropies[-1]["entropy"] - df_entropies[-1].loc[df_entropies[-1]["condition"].eq(normal),"entropy"].values[0]
    df_entropies[-1]["entropy_diff_dummy"] = df_entropies[-1]["entropy_dummy"] - df_entropies[-1].loc[df_entropies[-1]["condition"].eq(normal),"entropy_dummy"].values[0]
df_entropies = pd.concat(df_entropies,ignore_index=True)
# need to incorporate df_rate
df_entropies.set_index(["folder","condition"],inplace=True)
df_entropies['growth_rate'] = df_rates["growth_rate"]
df_entropies.reset_index(inplace=True)


# KL divergence and info entropy

# # replaced by block below
# for folder in df_entropies["folder"].unique():
#     plt.figure(figsize=(20,12))
#     g = sns.scatterplot(
#         data=df_entropies[df_entropies["folder"].eq(folder)],
#         x="index",y="KL_divergence",ci=None
#     )
#     g.set_xticks(df_entropies.loc[df_entropies["folder"].eq(folder),"index"])
#     g.set_xticklabels(df_entropies.loc[df_entropies["folder"].eq(folder),"condition"])
#     plt.savefig(f'plots/mutual_information_again/KL-divergence_{folder}.png')
#     plt.close()

for folder in df_entropies["folder"].unique():
    plt.figure()
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(folder),"index"],
        y=df_entropies.loc[df_entropies["folder"].eq(folder),"KL_divergence"],
        marker='x',label="Experiment"
    )
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(folder),"index"],
        y=df_entropies.loc[df_entropies["folder"].eq(folder),"KL_divergence_dummy"],
        marker='o',label="Random Shuffle"
    )
    plt.xticks(
        df_entropies.loc[df_entropies["folder"].eq(folder),"index"],
        df_entropies.loc[df_entropies["folder"].eq(folder),"condition"]
    )
    plt.legend()
    plt.xlabel("condition")
    plt.title("KL-divergence")
    plt.savefig(f'plots/mutual_information_again/KL-divergence_{folder}.png',dpi=600)
    plt.close()


w,h = figaspect(0.5)
plt.figure(figsize=(w,h))
for e,exp in enumerate(experiments.keys()):
    plt.scatter(
        x=df_entropies.loc[df_entropies['folder'].eq(experiments[exp]),"growth_rate"],
        y=df_entropies.loc[df_entropies['folder'].eq(experiments[exp]),"entropy"],
        marker="x",color=cmap(e) 
    )
    plt.scatter(
        x=df_entropies.loc[df_entropies['folder'].eq(experiments[exp]),"growth_rate"],
        y=df_entropies.loc[df_entropies['folder'].eq(experiments[exp]),"entropy_dummy"],
        marker="o",color=cmap(e),label=exp
    )
plt.scatter([],[],marker='x',color='k',label="Experiment")
plt.scatter([],[],marker='o',color='k',label="Random Shuffle")
plt.ylim(0,None)
plt.xlabel("Growth Rate")
plt.ylabel("Entropy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(f'plots/mutual_information_again/entropy_growthrate.png',dpi=600)
plt.close()


w,h = figaspect(0.5)
plt.figure(figsize=(w,h))
for e,exp in enumerate(experiments.keys()):
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"growth_rate","growth_rate"],
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"growth_rate","KL_divergence"],
        marker="x",color=cmap(e)
    )
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"growth_rate","growth_rate"],
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"growth_rate","KL_divergence_dummy"],
        marker="o",ax=g
    )
plt.scatter([],[],marker='x',color='k',label="Experiment")
plt.scatter([],[],marker='o',color='k',label="Random Shuffle")
plt.ylim(0,None)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(f'plots/mutual_information_again/klDivergence_growthrate.png')
plt.close()


w,h = figaspect(0.5)
plt.figure(figsize=(w,h))
for e,exp in enumerate(experiments.keys()):
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"KL_divergence"],
        y=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"entropy"],
        marker="x",color=cmap(e)
    )
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"KL_divergence_dummy"],
        y=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"entropy_dummy"],
        marker="o",color=cmap(e),edgecolor='white',
        label=exp
    )
plt.scatter([],[],marker='x',color='k',label="Experiment")
plt.scatter([],[],marker='o',color='k',label="Random Shuffle")
plt.ylim(0,None)
plt.xlabel("K-L Divergence")
plt.ylabel("Entropy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(f'plots/mutual_information_again/entropy_klDivergence.png',dpi=600)
plt.close()
# The question from the review 2 is because I did not make legends to
# explain the meanings of 'x' and 'o' markers. 
# The x markers are the real data (2% did have KL divergence of 0), 
# while o markers are random shuffles from the same group. 
# They should have a KL divergence from 2% experiment data, 
# and a higher information entropy, which are indeed what we see.


w,h = figaspect(0.5)
plt.figure(figsize=(w,h))
for e,exp in enumerate(experiments.keys()):
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"entropy"],
        y=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"KL_divergence"],
        marker="x",color=cmap(e)
    )
    plt.scatter(
        x=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"entropy_dummy"],
        y=df_entropies.loc[df_entropies["folder"].eq(experiments[exp]),"KL_divergence_dummy"],
        marker="o",color=cmap(e),edgecolor='white',
        label=exp
    )
plt.scatter([],[],marker='x',color='k',label="Experiment")
plt.scatter([],[],marker='o',color='k',label="Random Shuffle")
plt.ylim(0,None)
plt.xlabel("Entropy")
plt.ylabel("K-L Divergence")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(f'plots/mutual_information_again/klDivergence_entropy.png',dpi=600)
plt.close()


# glucose large experiment only:
w,h = figaspect(0.48)
plt.figure(figsize=(w,h))
for c,cond in enumerate(df_entropies.loc[df_entropies["folder"].eq("EYrainbow_glucose_largerBF"),"condition"].unique()):
    plt.scatter(
        x=df_entropies.loc[
            df_entropies["folder"].eq("EYrainbow_glucose_largerBF")
          & df_entropies["condition"].eq(cond),
            "KL_divergence"
        ],
        y=df_entropies.loc[
            df_entropies["folder"].eq("EYrainbow_glucose_largerBF")
          & df_entropies["condition"].eq(cond),
            "entropy"
        ],
        marker="x", color=cmap(list_colors["glucose"][c])
    )
    plt.scatter(
        x=df_entropies.loc[
            df_entropies["folder"].eq("EYrainbow_glucose_largerBF")
          & df_entropies["condition"].eq(cond),
            "KL_divergence_dummy"
        ],
        y=df_entropies.loc[
            df_entropies["folder"].eq("EYrainbow_glucose_largerBF")
          & df_entropies["condition"].eq(cond),
            "entropy_dummy"
        ],
        marker="o", color=cmap(list_colors["glucose"][c]),
        label=f"{cond*2/100}% w/v glucose"
    )
plt.scatter([],[],marker='x',color='k',label="Experiment")
plt.scatter([],[],marker='o',color='k',label="Random Shuffle")
plt.ylim(0,None)
plt.xlabel("K-L Divergence")
plt.ylabel("Entropy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(f'plots/mutual_information_again/entropy_klDivergence_glucose.png',dpi=600)
plt.close()


columns = ['cell length','cell area','cell volume',*properties]

for folder in subfolders:
    # mutual information
    # columns = ['condition','effective-length','cell-area','cell-volume',*properties]
    digitized = {}
    for column in columns:
        digitized[column] = np.digitize(
            df_corrcoef[column],
            bins=np.histogram(df_corrcoef[column],bins=100)[1]
        )
    num_col = len(columns)
    mx_mutualinfo = np.zeros((num_col,num_col))
    mx_miadjusted = np.zeros((num_col,num_col))
    mx_normalzied = np.zeros((num_col,num_col))
    for i,column0 in enumerate(columns):
        mx_mutualinfo[i,i] = metrics.mutual_info_score(digitized[column0],digitized[column0])
        mx_miadjusted[i,i] = 1.0
        mx_normalzied[i,i] = 1.0
        if (i+1)==num_col:
            break
        for j,column1 in enumerate(columns[i+1:]):
            mx_mutualinfo[i,i+j+1] = metrics.mutual_info_score(digitized[column0],digitized[column1])
            mx_miadjusted[i,i+j+1] = metrics.adjusted_mutual_info_score(digitized[column0],digitized[column1])
            mx_normalzied[i,i+j+1] = metrics.normalized_mutual_info_score(digitized[column0],digitized[column1])
    for matrix,mx_name in zip([mx_mutualinfo,mx_miadjusted,mx_normalzied],["mutual-info","adjusted-mutual-info","normalized-mutual-info"]):
        fig = px.imshow(
            matrix,
            x=columns,y=columns,
            color_continuous_scale="OrRd",range_color=[0,1]
        )
        fig.write_html(f'{Path("./plots/mutual_information_again")}/{mx_name}_{folder}.html')


# Correlation coefficient
for folder in subfolders:
    np_corrcoef = df_corrcoef.loc[df_corrcoef["folder"].eq(folder),columns].to_numpy()
    corrcoef = np.corrcoef(np_corrcoef,rowvar=False)
    fig = px.imshow(
            corrcoef,
            x=columns,y=columns,
            color_continuous_scale = "RdBu_r",range_color=[-1,1]
        )
    fig.update_layout(
        # font_family="Arial Black",
        font_size=14
    )
    # fig.write_html(f'{Path("./data/correlation")}/corrcoef_{folder}.html')
    fig.write_html(f'{Path("./data/REBUTTAL_PLOTS/correlation")}/corrcoef-nocond_{folder}.html')
    
    # trivial visualization change:
    # only keep lower half
    rows,cols = corrcoef.shape
    corrcoef_triangle_low = corrcoef.copy()
    for r in range(rows-1):
        for c in range(r+1,cols):
            corrcoef_triangle_low[r,c] = 0
    fig = px.imshow(
            corrcoef_triangle_low,
            x=columns,y=columns,
            color_continuous_scale = "RdBu_r",range_color=[-1,1]
        )
    fig.update_layout(
        # font_family="Arial Black",
        font_size=14
    )
    fig.write_html(f'{Path("./data/REBUTTAL_PLOTS/correlation")}/corrcoef-nocond_{folder}_lower.html')
    # only keep upper half
    corrcoef_triangle_upp = corrcoef.copy()
    for c in range(cols-1):
        for r in range(c+1,rows):
            corrcoef_triangle_upp[r,c] = 0
    fig = px.imshow(
            corrcoef_triangle_upp,
            x=columns,y=columns,
            color_continuous_scale = "RdBu_r",range_color=[-1,1]
        )
    fig.update_layout(
        # font_family="Arial Black",
        font_size=14
    )
    fig.update_xaxes(side="top")
    fig.update_yaxes(side="right")
    fig.write_html(f'{Path("./data/REBUTTAL_PLOTS/correlation")}/corrcoef-nocond_{folder}_upper.html')


    # for condi in df_corrcoef.loc[df_corrcoef["folder"].eq(folder),"condition"].unique():
    #     np_corrcoef = df_corrcoef.loc[(df_corrcoef["folder"].eq(folder) & df_corrcoef['condition'].eq(condi)),columns].to_numpy()
    #     corrcoef = np.corrcoef(np_corrcoef,rowvar=False)
    #     fig = px.imshow(
    #             corrcoef,
    #             x=columns,y=columns,
    #             color_continuous_scale="RdBu_r",range_color=[-1,1]
    #         )
    #     fig.write_html(f'{Path("./data/correlation/rebuttal_202401")}/groupby_conditions/corrcoef-nocond_{folder}_{str(condi).replace(".","-")}.html')

    # # Pairwise relation atlas, super slow!
    # fig_pair = sns.PairGrid(df_corrcoef,hue="condition",vars=['cell length','cell area','cell volume',*properties],height=3.0)
    # fig_pair.map_diag(sns.histplot)
    # # f_pairig.map_offdiag(sns.scatterplot)
    # fig_pair.map_upper(sns.scatterplot)
    # fig_pair.map_lower(sns.kdeplot)
    # fig_pair.add_legend()
    # fig_pair.savefig(f'{Path("./data/correlation")}/pairplot_{folder}.png')

    # # Pairwise relation individual
    # sns.set_palette(sns.blend_palette(['red','blue']))
    # g = sns.jointplot(
    #                   data=df_corrcoef, 
    #                   x="cell-volume", y="total-fraction-mitochondria",
    #                   hue="condition", kind="kde"
    #     )
    # # g.plot_joint(sns.kdeplot, zorder=0, levels=6)
    # g.plot_marginals(sns.rugplot, height=-.15, clip_on=False)


# EXPERIMENT CONDITION LEVEL

pivot_bycondition = df_bycell.groupby(['folder','organelle','condition']).mean()[['mean','count','total','cell-area','cell-volume','total-fraction']]
pivot_bycondition["cell_count"] = df_bycell[['folder','organelle','condition','mean']].groupby(['folder','organelle','condition']).count()
df_bycondition = pivot_bycondition.reset_index()

# Growth Rates
# old data taken together with imaging
df_rates = pd.read_csv(str(Path("./plots/growthrate")/"growth_rate.csv"))
df_rates.rename(columns={"experiment":"folder"},inplace=True)
df_rates.set_index(["folder","condition"],inplace=True)

# new data taken from deliberate replicates
df_rate_replicate = pd.read_csv("plots/growthrate/rebuttal_OD_replicate.csv")
df_rate_replicate["time"] = pd.to_datetime(df_rate_replicate["time"],format="%H:%M")
first_times = df_rate_replicate[["group","replicate","time"]].groupby(["group","replicate"]).first()
first_ODs = df_rate_replicate[["group","replicate","OD"]].groupby(["group","replicate"]).first()
df_rate_replicate.set_index(["group","replicate"],inplace=True)
df_rate_replicate["first_times"] = first_times
df_rate_replicate["first_ODs"] = first_ODs
df_rate_replicate["minute"] = (df_rate_replicate["time"] - df_rate_replicate["first_times"]).dt.total_seconds()/60.0
df_rate_replicate["normalized"] = df_rate_replicate["OD"] / df_rate_replicate["first_ODs"]
df_rate_replicate.reset_index(inplace=True)
# not tested, not even remember what interfaces this new dataset gives

df_entropies.set_index(["folder","condition"],inplace=True)
df_entropies.loc[:,"growth_rate"] = df_rates["growth_rate"]
df_entropies.reset_index(inplace=True)

df_bycondition.set_index(["folder","condition"],inplace=True)
df_bycondition.loc[:,"growth_rate"] = df_rates["growth_rate"]
df_bycondition.reset_index(inplace=True)

df_bycondition.set_index(["folder","condition"],inplace=True)
idx_fraction_bycondition = df_bycondition[df_bycondition["organelle"].eq("ER")].index
df_fraction_bycondition  = pd.DataFrame(index=idx_fraction_bycondition)
for orga in [*organelles,"non-organelle"]:
    df_fraction_bycondition[orga] = df_bycondition.loc[df_bycondition["organelle"].eq(orga),"total-fraction"]
df_fraction_bycondition["growth-rate"] = df_bycondition.loc[df_bycondition["organelle"].eq("ER"),"growth_rate"]
df_fraction_bycondition.reset_index(inplace=True)
df_bycondition.reset_index(inplace=True)
# df_fraction_bycondition.to_csv(str(Path("./data/fraction_rate")/"fraction-by-conditions.csv"),index=False)

df_bycell.set_index(["folder","condition"],inplace=True)
df_bycell["growth_rate"] = df_rates["growth_rate"]
df_bycell.reset_index(inplace=True)

df_bycell.set_index(["folder","condition","field","idx-cell"],inplace=True)
idx_fraction_bycell = df_bycell[df_bycell["organelle"].eq("ER")].index
df_fraction_bycell  = pd.DataFrame(index=idx_fraction_bycell)
for orga in [*organelles,"non-organelle"]:
    df_fraction_bycell[orga] = df_bycell.loc[df_bycell["organelle"].eq(orga),"total-fraction"]
df_fraction_bycell["cell-volume"] = df_bycell.loc[df_bycell["organelle"].eq("ER"),"cell-volume"]
df_fraction_bycell["growth-rate"] = df_bycell.loc[df_bycell["organelle"].eq("ER"),"growth_rate"]
df_fraction_bycell.reset_index(inplace=True)
df_bycell.reset_index(inplace=True)
# df_fraction_bycell.to_csv(str(Path("./data/fraction_rate")/"fraction-by-cells.csv"),index=False)


# plot volume fraction vs. growth rate
fig = px.line(
    df_bycondition.loc[df_bycondition['organelle'].eq("non-organelle")],
    x='growth_rate',y='total',
    color="folder"
)
fig.write_html(str(Path("./data/growthrate")/"non-organelle-vol-total_growth-rate.html"))

df_bycondition = df_bycondition[df_bycondition["folder"].isin(exp_folder)]
# df_bycondition.to_csv(str(Path("./data/fraction_rate")/"to_plot.csv"))
plt.figure(figsize=(10,8))
sns.scatterplot(
    data=df_bycondition,
    x="growth_rate",y="total-fraction",
    hue="folder",style="organelle",s=25
)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(
    str(Path("./data/fraction_rate")/"fraction_rate_cyto_all.png"),
    bbox_inches='tight')
plt.close()

for orga in [*organelles,"non-organelle"]:
    plt.figure(figsize=(10,8))
    sns.scatterplot(
        data=df_bycondition[df_bycondition["organelle"].eq(orga)],
        x="growth_rate",y="total-fraction",
        hue="folder",s=25
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.savefig(
        str(Path("./data/fraction_rate")/f"fraction_rate_cyto_{orga}.png"),
        bbox_inches='tight'
    )
    plt.close()
    


# PLOTS

def plot_histo_violin(df,prop_y,prop_x="cell_area",savepath="histo-violin.html"):
    df["plotly"] = df[prop_y].rank(pct=True) # `pct` works but fuck why
    fig = make_subplots(rows=1,cols=1)
    fig.append_trace(
        go.Violin(
            y=df["plotly"],
            # y=np.arange(len(df[prop_y].unique())),
            x=df[prop_x]
        ),
        row=1, col=1
    )
        
    wid = 2./len(df["plotly"].unique())
    
    fig.update_traces(orientation='h', side='positive', width=wid, points="all",pointpos=0.5,jitter=0.4,line_color="deepskyblue",marker_color="tomato",marker_opacity=0.1,row=1, col=1)
    # print(df[prop_y].unique())
    # print(df["plotly"].unique())
    fig.update_layout(
                    title=Path(savepath).stem.replace("_"," "),
                    yaxis_title=prop_y,
                    yaxis_ticktext=df[prop_y].unique(),
                    yaxis_tickvals=df["plotly"].unique(),
                    xaxis_title=f'{prop_x}'
    )
    fig.show()
    fig.write_html(str(savepath))
    return None

def plot_histo_box(df,prop_y,prop_x="cell_area",savepath=""):
    fig = make_subplots(rows=1,cols=1)
    for prop in df[prop_y].unique():
        fig.append_trace(
            go.Box(x=df.loc[df[prop_y]==prop,prop_x],
                   boxmean='sd'),
            row=1, col=1
        )
    
    fig.update_layout(
                    title=Path(savepath).stem.replace("_"," "),
                    yaxis_title=prop_y,
                    yaxis_ticktext=df[prop_y].unique(),
                    yaxis_tickvals=df[prop_y].unique(),
                    xaxis_title=f'{prop_x}'
    )
    fig.show()
    fig.write_html(str(savepath))
    return None

for folder in subfolders:
    for orga in organelles:
        plot_histo_violin(
            df_bycell.loc[df_bycell["organelle"].eq(orga) & df_bycell["folder"].eq(folder)]
            ,"condition","mean",
            f'{Path("./data/figures")}/cellular_mean_vs_condition_{folder}_{orga}.html'
        )
        plot_histo_violin(
            df_bycell.loc[df_bycell["organelle"].eq(orga) & df_bycell["folder"].eq(folder)],
            "condition","count",
            f'{Path("./data/figures")}/cellular_count_vs_condition_{folder}_{orga}.html'
        )


def plot_group_hist(df,prop_y,prop_x="cell_area",savepath=""):
    fig = make_subplots(rows=1,cols=1)
    for i,prop in enumerate(df[prop_y].unique()):
        fig.append_trace(
            go.Histogram(
                x=df.loc[df[prop_y].eq(prop),prop_x],
                name=df[prop_y].unique()[i]),
            row=1, col=1
        )
    
    fig.update_layout(
                    barmode='overlay',
                    title=Path(savepath).stem.replace("_"," "),
                    yaxis_title=prop_y,
                    xaxis_title=f'{prop_x}',
                    bargap=0.001,bargroupgap=0.05
    )
    fig.update_traces(opacity=0.75)
    # fig.show()
    fig.write_html(str(savepath))
    return None

for folder in subfolders:
    for orga in organelles:
        plot_group_hist(
            df_bycell.loc[df_bycell["organelle"].eq(orga) & df_bycell["folder"].eq(folder)],
            "condition","mean",
            f'{Path("./data/figures")}/organelle_mean_vs_condition_{folder}_{orga}.html'
        )

# PCA 
def make_pca_plots(experiment,property,groups=None,non_organelle=False,has_volume=False,
                    is_normalized=True,divided_by_max=False,divided_by_median=False,
                    saveto="./plots/"):
    for pca_subfolder in [ "pca_data/",
                           "pca_compare/",
                           "pca_projection_extremes/",
                           "pca_projection_all_plt/",
                           "pca_projection_all/"
                         ]:
        if saveto is not None and not (Path(saveto)/pca_subfolder).exists():
            (Path(saveto)/pca_subfolder).mkdir()
        continue
    folder = experiments[experiment]
    name = f"{'all-conditions' if groups is None else 'extremes'}_{'has-cytoplasm' if non_organelle else 'no-cytoplasm'}_{'cell-volume' if has_volume else 'organelle-only'}_{property}_{'norm-mean-std' if is_normalized else 'raw'}"
    print("PCA Anaysis: ",folder,name)

    df_orga_perfolder = df_bycell[df_bycell["folder"].eq(folder)]
    df_orga_perfolder.set_index(["condition","field","idx-cell"],inplace=True)
    idx = df_orga_perfolder.groupby(["condition","field","idx-cell"]).count().index
    
    columns = [*organelles,"non-organelle"] if non_organelle else organelles
    df_pca = pd.DataFrame(index=idx,columns=columns)
    num_pc = 7 if non_organelle else 6
    
    for orga in columns:
        df_pca[orga] = df_orga_perfolder.loc[df_orga_perfolder["organelle"].eq(orga),property]
    
    if has_volume:
        df_pca["cell-volume"] = df_orga_perfolder.loc[df_orga_perfolder["organelle"].eq("ER"),"cell volume"]
        columns = ["cell volume",*columns]
        num_pc += 1
    
    if divided_by_max:
        for col in columns:
            df_pca[col] = df_pca[col] / df_pca[col].max()
    if divided_by_median:
        for col in columns:
            df_pca[col] = df_pca[col] / df_pca[col].median()        

    if is_normalized:
        for col in columns:
            df_pca[col] = (df_pca[col]-df_pca[col].mean())/df_pca[col].std()
    
    df_pca.reset_index(inplace=True)

    if saveto is None:
        return df_pca

    # Find the the direction of the condition change:
    df_centroid = df_pca.groupby("condition")[columns].mean()
    fitter_centroid = LinearRegression(fit_intercept=False)
    np_centroid = df_centroid.to_numpy()
    fitter_centroid.fit(np_centroid,np.ones(np_centroid.shape[0]))
    
    vec_centroid_start = df_centroid.loc[groups[0],:].to_numpy()
    vec_centroid_start[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_start[:-1]))/fitter_centroid.coef_[-1]

    vec_centroid_ended = df_centroid.loc[groups[-1],:].to_numpy()
    vec_centroid_ended[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_ended[:-1]))/fitter_centroid.coef_[-1]

    vec_centroid = vec_centroid_ended - vec_centroid_start
    vec_centroid = vec_centroid/np.linalg.norm(vec_centroid)
    np.savetxt(str(Path(saveto)/f"pca_data/condition-vector_{folder}_{name}.txt"),vec_centroid)

    # Get Principal Components (PCs)
    np_pca = df_pca[columns].to_numpy()
    pca = PCA(n_components=num_pc)
    pca.fit(np_pca)
    pca_components = pca.components_
    pca_var_ratios = pca.explained_variance_ratio_

    # Calculate cos<condition,PCs>, and realign the PCs
    cosine_pca = np.dot(pca_components,vec_centroid)
    for c in range(len(cosine_pca)):
        if cosine_pca[c] < 0:
            pca_components[c] = -pca_components[c]
    cosine_pca = np.abs(cosine_pca)
    # save and plot PCs without sorting.
    np.savetxt(f'{saveto}/pca_data/cosine_{folder}_{name}.txt',cosine_pca)
    np.savetxt(f'{saveto}/pca_data/pca-components_{folder}_{name}.txt',pca_components)
    np.savetxt(f'{saveto}/pca_data/pca-explained-ratio_{folder}_{name}.txt',pca_var_ratios)
    fig_components = px.imshow(
        pca_components,
        x=columns, y=[f"PC{i}" for i in range(num_pc)],
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0
    )
    fig_components.write_html(f'{saveto}/pca_data/pca_components_{folder}_{name}.html')

    # Sort PCs according to the cosine
    arg_cosine = np.argsort(cosine_pca)[::-1]
    pca_components_sorted = pca_components[arg_cosine]
    # save and plot the PCs with sorting
    np.savetxt(f'{saveto}/pca_compare/condition-sorted-index_{folder}_{name}.txt',arg_cosine)
    np.savetxt(f'{saveto}/pca_compare/condition-sorted-cosine_{folder}_{name}.txt',cosine_pca[arg_cosine])
    np.savetxt(f'{saveto}/pca_compare/condition-sorted-pca-components_{folder}_{name}.txt',pca_components_sorted)
    fig_components_sorted = px.imshow(
        pca_components_sorted,
        x=columns, y=[f"PC{i}" for i in arg_cosine],
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0
    )
    fig_components_sorted.write_html(f'{saveto}/pca_compare/condition-sorted-pca_components_{folder}_{name}.html')
    # plot the cosine 
    plt.figure(figsize=(15,12))
    plt.barh(np.arange(num_pc),cosine_pca[arg_cosine[::-1]],align='center')
    plt.yticks(np.arange(num_pc),[f"PC{i}" for i in arg_cosine[::-1]])
    plt.xlabel(r"$cos\left<condition\ vector,PC\right>$")
    plt.title(f"{folder}")
    plt.savefig(f'{saveto}/pca_compare/condition-sorted-cosine_{folder}_{name}.png')
    plt.close()


    # Draw projections onto the PCs
    for i_pc in range(len(pca_components)):
        base = pca_components[i_pc]
        df_pca[f"proj{i_pc}"] = df_pca.apply(lambda x:np.dot(base,x.loc[columns]),axis=1)
    pc2proj = arg_cosine[:3]
    df_pca_extremes = df_pca.loc[df_pca["condition"].isin(groups)]

    # 3d projection
    figproj = plt.figure(figsize=(15,12))
    ax = figproj.add_subplot(projection="3d")
    for condi in groups[::-1]:
        pc_x = df_pca_extremes.loc[df_pca_extremes["condition"].eq(condi),f"proj{pc2proj[0]}"],
        pc_y = df_pca_extremes.loc[df_pca_extremes["condition"].eq(condi),f"proj{pc2proj[1]}"],
        pc_z = df_pca_extremes.loc[df_pca_extremes["condition"].eq(condi),f"proj{pc2proj[2]}"],
        ax.scatter(
            pc_x, pc_y, pc_z,
            s=55,alpha=0.3,label=f"{condi}"
        )
    ax.set_xlabel(f"proj {pc2proj[0]}")
    ax.set_ylabel(f"proj {pc2proj[1]}")
    ax.set_zlabel(f"proj {pc2proj[2]}")
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[0]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.set_ylim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[1]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.set_zlim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[2]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
    ax.legend(loc=(1.04,1.0))
    figproj.savefig(f'{saveto}/pca_projection_extremes/pca_projection3d_{folder}_{name}_pc{"".join([str(p) for p in pc2proj])}.png')
    plt.close(figproj)

    # 3d projections, all conditions
    for d,condi in enumerate(np.sort(df_pca["condition"].unique())):
        if condi == groups[1]:
            continue
        figproj = plt.figure(figsize=(15,12))
        ax = figproj.add_subplot(projection="3d")
        
        pc_x = df_pca.loc[df_pca["condition"].eq(groups[-1]),f"proj{pc2proj[0]}"],
        pc_y = df_pca.loc[df_pca["condition"].eq(groups[-1]),f"proj{pc2proj[1]}"],
        pc_z = df_pca.loc[df_pca["condition"].eq(groups[-1]),f"proj{pc2proj[2]}"],
        ax.scatter(
            pc_x, pc_y, pc_z,
            edgecolor='white',facecolor=sns.color_palette('tab10')[0],
            s=55,alpha=0.3,label=f"{groups[-1]}"
        )
        
        pc_x = df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[0]}"],
        pc_y = df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[1]}"],
        pc_z = df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[2]}"],
        ax.scatter(
            pc_x, pc_y, pc_z,
            edgecolor='white',facecolor=sns.color_palette('tab10')[list_colors[experiment][d]],
            s=55,alpha=0.3,label=f"{condi}"
        )
        ax.set_xlabel(f"proj {pc2proj[0]}")
        ax.set_ylabel(f"proj {pc2proj[1]}")
        ax.set_zlabel(f"proj {pc2proj[2]}")
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[0]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
        ax.set_ylim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[1]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
        ax.set_zlim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[2]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
        ax.legend(loc=(1.04,0.5))
        figproj.savefig(f'{saveto}/pca_projection_all_plt/pca_projection3d_{folder}_{name}_condi-{str(condi).replace(".","-")}_pc{"".join([str(p) for p in pc2proj])}.png')
        plt.close(figproj)
 
    # 2d projections
    sns.set_style("whitegrid")
    for first,second in ((0,1),(0,2),(1,2)):
        plt.figure(figsize=(15,12))
        sns_plot = sns.scatterplot(
            data=df_pca_extremes[df_pca_extremes["condition"].eq(groups[0])],
            x=f"proj{pc2proj[first]}",y=f"proj{pc2proj[second]}",
            color=sns.color_palette("tab10")[1],s=100,alpha=0.75
        )
        sns_plot = sns.scatterplot(
            data=df_pca_extremes[df_pca_extremes["condition"].eq(groups[1])],
            x=f"proj{pc2proj[first]}",y=f"proj{pc2proj[second]}",
            color=sns.color_palette("tab10")[0],s=100,alpha=0.75
        )
        sns_plot.figure.savefig(f'{saveto}/pca_projection_extremes/pca_projection2d_{folder}_{name}_pc{pc2proj[first]}{pc2proj[second]}.png')
        plt.close()

    # 2d projections, all conditions
    for d,condi in enumerate(np.sort(df_pca["condition"].unique())):
        if condi == groups[1]:
            continue
        for first,second in ((0,1),(0,2),(1,2)):
            plt.figure(figsize=(15,12))
            sns_plot = sns.scatterplot(
                data=df_pca[df_pca["condition"].eq(groups[1])],
                x=f"proj{pc2proj[first]}",y=f"proj{pc2proj[second]}",
                color=sns.color_palette("tab10")[0],s=100,alpha=0.75
            )
            sns_plot = sns.scatterplot(
                data=df_pca[df_pca["condition"].eq(condi)],
                x=f"proj{pc2proj[first]}",y=f"proj{pc2proj[second]}",
                color=sns.color_palette("tab10")[list_colors[experiment][d]],s=100,alpha=0.75
            )
            plt.xlim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[first]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
            plt.ylim(*(np.percentile(df_pca_extremes[f"proj{pc2proj[second]}"].to_numpy(),[1,99])+np.array([-0.1,0.1])))
            sns_plot.figure.savefig(f'{saveto}/pca_projection_all_plt/pca_projection2d_{folder}_{name}_condi-{str(condi).replace(".","-")}_pc{pc2proj[first]}{pc2proj[second]}.png')
            plt.close()

    # draw interactive HTML with Plotly:
    figproj = go.Figure()
    for d,condi in enumerate(np.sort(df_pca["condition"].unique())):
        figproj.add_trace(
            go.Scatter3d(
                x=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[0]}"],
                y=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[1]}"],
                z=df_pca.loc[df_pca["condition"].eq(condi),f"proj{pc2proj[2]}"],
                name=condi,
                mode="markers",
                marker=dict(
                            size=4,
                            color=f"rgb{tuple((np.array(sns.color_palette('tab10')[list_colors[experiment][d]])*255).astype(int))}",
                            opacity=0.8
                       )
            )
        )
    figproj.update_layout(
        scene=dict(
            xaxis=dict(
                title=f"proj{pc2proj[0]}",
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='grey',
                showline = True,
                zeroline=True,
                zerolinecolor='black'
            ),
            yaxis=dict(
                title=f"proj{pc2proj[1]}",
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='grey',
                showline=True,
                zeroline=True,
                zerolinecolor='black'
            ),
            zaxis=dict(
                title=f"proj{pc2proj[2]}",
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='grey',
                showline = True,
                zeroline=True,
                zerolinecolor='black'
            )
        )
    )
    figproj.write_html(f'{Path(f"{saveto}/pca_projection_all")}/pca_projection3d_{folder}_{name}_pc{"".join([str(p) for p in pc2proj])}.html')
    return df_pca

for experiment in exp_names:
    make_pca_plots(experiment,"total-fraction",groups=extremes[experiments[experiment]],has_volume=False,is_normalized=True,non_organelle=False,saveto="plots/PCA_interactive_same_colors")

saveto="plots/PCA_interactive_same_colors"

dict_pc        = {}
dict_cosine    = {}
dict_var_ratio = {}
for expm in exp_names: 
    file_pc = f"{saveto}/pca_data/pca-components_{experiments[expm]}_extremes_no-cytoplasm_organelle-only_total-fraction_norm-mean-std.txt"
    file_cosine = f"{saveto}/pca_data/cosine_{experiments[expm]}_extremes_no-cytoplasm_organelle-only_total-fraction_norm-mean-std.txt"
    file_var_ratio = f"{saveto}/pca_data/pca-explained-ratio_{experiments[expm]}_extremes_no-cytoplasm_organelle-only_total-fraction_norm-mean-std.txt"
    dict_pc[expm] = np.loadtxt(file_pc)
    dict_cosine[expm] = np.loadtxt(file_cosine)
    dict_var_ratio[expm] = np.loadtxt(file_var_ratio)

# find PCs in different experiments most similar to glucose PCs
dict_product = {}
dict_ranking = {}
for expm in exp_names: 
    list_product = []
    list_ranking = []
    for i in range(6):
        product = np.dot(dict_pc[expm],dict_pc["glucose"][i])
        indice_pc = np.argmax(np.abs(product))
        list_product.append(product[indice_pc])
        list_ranking.append(indice_pc)
    dict_product[expm] = list_product
    dict_ranking[expm] = [f"PC{r}" for r in list_ranking]
glu2similar = pd.DataFrame(dict_product).to_numpy()
glu2ranking = pd.DataFrame(dict_ranking).to_numpy()
fig_glu2others = px.imshow(
    glu2similar,
    x=exp_names, y=None,
    color_continuous_scale="RdBu_r",color_continuous_midpoint=0
)
fig_glu2others.update_traces(text=glu2ranking,texttemplate="%{text}")
fig_glu2others.update_xaxes(side="top")
fig_glu2others.write_html(f'{saveto}/pca_compare/compare2glucose.html')


# summary of experiments
sq_summary = np.empty((len(exp_names),len(exp_names)))
sq_summary[:] = np.nan
for i0,expm0 in enumerate(exp_names):
    for i1,expm1 in enumerate(exp_names[i0:]):
        sq_products = np.zeros((6,6)) # hard-coded num of organelles
        for s0 in range(6):
            for s1 in range(s0,6):
                sq_products[s0,s1] = dict_cosine[expm0][s0]*np.dot(dict_pc[expm0][s0],dict_pc[expm1][s1])*dict_cosine[expm1][s1]
        sq_summary[i0,i0+i1] = np.sum(sq_products)
fig_summary = px.imshow(
    sq_summary,
    x=exp_names,y=exp_names,
    color_continuous_scale="RdBu_r",color_continuous_midpoint=0
)
fig_summary.write_html(f'{saveto}/pca_compare/summary_7b18c1c_T.html')


# superplot (a box and whisker with the individual datapoints overlaid on top)
# of normalized volume fractions for each organelle for each set of perturbations
# (for each organelle make a plot that overlays all the glucose normalized vol frac distributions, 
# for each organelle make another plot that overlays all the leucine normalized vol frac distributions, etc) 
for experiment in experiments:
    df_pca = make_pca_plots(
        experiment=experiment,
        property="total-fraction",
        groups=None,
        has_volume=False,
        is_normalized=True,
        non_organelle=False,
        saveto=None
    )
    for organelle in organelles:
        normalizeds = []
        conditions  = []
        colors = []
        for c,condition in enumerate(np.sort(df_pca["condition"].unique())):
            normalizeds.append(df_pca.loc[df_pca["condition"].eq(condition),organelle])
            conditions.append(converts[experiment](condition))
            colors.append(sns.color_palette("tab10")[list_colors["glucose"][c]])
        
        fig, ax = plt.subplots()
        bplot = ax.boxplot(
                    normalizeds,
                    showfliers=False,
                    patch_artist=True,
                    labels=conditions,
                    medianprops={"color":'k'}
                )
        for (patch,color) in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # for n,normalized in enumerate(normalizeds):
        #     ax.scatter(
        #         n + 1 - 0.15 + 0.3*np.random.random(size=len(normalized)),
        #         normalized,
        #         s=2,color=colors[n],alpha=0.3
        #     )
        ax.set_title(f"{' '.join([s if s.isupper() else s.title() for s in experiment.split()])} Perturbation\n{organelle if organelle.isupper() else organelle.title()}")
        ax.set_ylabel("Normalized Volume Fraction")
        ax.set_xlabel(f"{perturbations[experiment]} ({units[experiment]})")
        plt.savefig(f"plots/normalized_volume_fraction/boxOnly_normalizedVolFraction_{experiment}_{organelle}.png",dpi=600)

# power law
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# # Shankar did not give me the glu-0.5 data
# df_tmp = df_bycell.loc[
#     df_bycell["folder"].eq("EYrainbow_glucose_largerBF")
#     &df_bycell["organelle"].eq("non-organelle")
#     &df_bycell["condition"].eq(0.5)
#     ,
#     ["total","cell-volume"]
# ]
# df_tmp = np.log(df_tmp)
# df_tmp = df_tmp[df_tmp["total"]>2]
# df_tmp.to_csv(
#     'data/power_law/cellvolvscytovol_glucoselargerBF_0-5_loglog.csv',
#     header=False,index=False
# )

# dfs = []
# for filepath in Path("data/power_law/").glob("*_loglog.csv"):
#     if "scambled" in filepath.stem:
#         print(filepath)
#         df_scambled = pd.read_csv(str(filepath),names=["log-Vcyto",r"log-Vcell"])
#         df_scambled.loc[df_scambled["log-Vcyto"].astype(str).str.contains("i"),"log-Vcyto"] = np.nan
#         df_scambled["log-Vcyto"] = df_scambled["log-Vcyto"].astype(float)
#         continue
#     df = pd.read_csv(str(filepath),names=["log-Vcyto",r"log-Vcell"])
#     condition = float(filepath.stem.partition("BF_")[2].partition("_")[0].replace('-','.'))
#     df.loc[df["log-Vcyto"].eq(-np.inf),"log-Vcyto"] = np.nan
#     df.loc[df["log-Vcyto"].astype(str).str.contains("i"),"log-Vcyto"] = np.nan
#     df["log-Vcyto"] = df["log-Vcyto"].astype(float)
#     df["condition"] = condition
#     dfs.append(df)
# dfs = pd.concat(dfs,ignore_index=True)

dfs = df_bycell.loc[
                    df_bycell["folder"].eq("EYrainbow_glucose_largerBF")
                  & df_bycell["organelle"].eq("ER"),
                    ["condition","cell-volume"]
                ]
dfs = dfs.reset_index().drop("index",axis=1)
dfs["cell-volume"] = np.log(dfs["cell-volume"])
for orga in [*organelles,"non-organelle"]:
    dfs[orga] = np.log(
                        df_bycell.loc[
                                      df_bycell["folder"].eq("EYrainbow_glucose_largerBF") 
                                    & df_bycell["organelle"].eq(orga),"total"
                                ].reset_index().drop("index",axis=1)
                    )
dfs.replace([np.inf, -np.inf], np.nan, inplace=True)
dfs.dropna(axis=0,inplace=True)

x = dfs["non-organelle"].values.reshape(-1,1)
y = dfs["cell-volume"].values
fitter.fit(x,y)
print("without exclusion", f"pooled" ,fitter.coef_, fitter.intercept_)
# >>> without exclusion pooled [0.42146724] 3.5076067960600694
dfs = dfs[dfs['non-organelle']>2]
x = dfs["non-organelle"].values.reshape(-1,1)
y = dfs["cell-volume"].values
fitter.fit(x,y)
print("with exclusion", f"pooled" ,fitter.coef_, fitter.intercept_)
# >>> with exclusion pooled [0.50221085] 3.142924975964997

x = dfs["non-organelle"].values
y = dfs["cell-volume"].values
res = scipy.odr.ODR(
                    scipy.odr.Data(x,y),
                    scipy.odr.unilinear
                ).run().beta[0]
print("without exclusion", f"symmetric" , res)
# >>> without exclusion symmetric 0.45422110021931483
threhold1 = np.percentile(dfs['non-organelle'],10) 
threhold2 = np.percentile(dfs['cell-volume'],10) 
dfs = dfs[dfs['non-organelle']>threhold1]
dfs = dfs[dfs['cell-volume']>threhold2]
x = dfs["non-organelle"].values
y = dfs["cell-volume"].values
res = scipy.odr.ODR(
                    scipy.odr.Data(x,y),
                    scipy.odr.unilinear
                ).run().beta[0]
print("with exclusion", f"symmetric" , res)


fitter = LinearRegression()
for cond in np.sort(dfs["condition"].unique()):
    x = dfs.loc[dfs["condition"].eq(cond),"non-organelle"].values.reshape(-1,1)
    y = dfs.loc[dfs["condition"].eq(cond),"cell-volume"].values
    fitter.fit(x,y)
    print("with exclusion", f"{cond*2/100}% glucose" ,fitter.coef_, fitter.intercept_)
# >>> without exclusion 0.0% glucose [0.36748129] 3.8123584248286857
# >>> without exclusion 0.01% glucose [0.48904931] 3.1963293062829417
# >>> without exclusion 0.1% glucose [0.45168254] 3.447715116047131
# >>> without exclusion 1.0% glucose [0.65527128] 2.2475956042518463
# >>> without exclusion 2.0% glucose [0.6174451] 2.477665894757461
# >>> without exclusion 4.0% glucose [0.66546864] 2.260198088347158
# >>> 
# >>> with exclusion 0.0% glucose [0.44245338] 3.4978532063642183
# >>> with exclusion 0.01% glucose [0.61766572] 2.607062652666553
# >>> with exclusion 0.1% glucose [0.57374911] 2.913059839803107
# >>> with exclusion 1.0% glucose [0.708232] 1.988237369816059
# >>> with exclusion 2.0% glucose [0.69546985] 2.1050524593477573
# >>> with exclusion 4.0% glucose [0.66886946] 2.243887408369273

df_scambled = dfs[dfs["condition"].eq(100)]
df_scambled["cell-volume"] = np.random.permutation(df_scambled["cell-volume"])


# Plot
plt.rcParams['font.size'] = '16'
fig,ax = plt.subplots()
for i,condi in enumerate(np.sort(dfs["condition"].unique())):
    print(i,condi,list_colors["glucose"][i])
    # ax.axis("equal")
    ax.scatter(
        x=dfs.loc[dfs["condition"].eq(condi),"non-organelle"],
        y=dfs.loc[dfs["condition"].eq(condi),"cell-volume"],
        s=12,
        label=f"{condi/100*2}% glucose",
        color=sns.color_palette("tab10")[list_colors["glucose"][i]],alpha=0.6,edgecolors='w'
    )
    ax.set_adjustable('datalim')
inlet = fig.add_axes([0.65,0.25,0.25,0.25])
inlet.axis("equal")
inlet.set_xbound(1.5,7.5)
inlet.set_ybound(3,8)
inlet.scatter(
    x=df_scambled["non-organelle"],
    y=df_scambled["cell-volume"],
    s=12,
    label="scambled",
    color="grey",alpha=0.5,edgecolors='w'
)
ax.set_adjustable('datalim')
x_min,x_max = dfs["non-organelle"].min(),dfs["non-organelle"].max()
y_min,y_max = dfs["cell-volume"].min(),dfs["cell-volume"].max()
ax.plot(
    [x_min,x_max],[y_max-(x_max-x_min)+1.5,y_max+1.5],"k--"
)
ax.plot(
    [x_min,x_max],[y_max-(2/3)*(x_max-x_min),y_max],"r--"
)
ax.set_xlabel(r"$log(V_{cyto})$")
ax.set_ylabel(r"$log(V_{cell})$")
# ax.legend()

plt.savefig("plots/power_law/power_law_rectangular_lowincluded.png",dpi=600)
plt.close()


# exclude the cells with too little non-organelle volumes
dfs = dfs[dfs['non-organelle']>2]
df_scambled = dfs[dfs["condition"].eq(100)]
df_scambled["cell-volume"] = np.random.permutation(df_scambled["cell-volume"])

plt.rcParams['font.size'] = '16'
fig,ax = plt.subplots()
for i,condi in enumerate(np.sort(dfs["condition"].unique())):
    print(i,condi,list_colors["glucose"][i])
    # ax.axis("equal")

    ax.set_xlim(2,7.5)
    ax.set_ylim(3,7.5)
    ax.scatter(
        x=dfs.loc[dfs["condition"].eq(condi),"non-organelle"],
        y=dfs.loc[dfs["condition"].eq(condi),"cell-volume"],
        s=12,
        label=f"{condi/100*2}% glucose",
        color=sns.color_palette("tab10")[list_colors["glucose"][i]],alpha=0.6,edgecolors='w'
    )
    ax.set_adjustable('datalim')
inlet = fig.add_axes([0.65,0.25,0.25,0.25])
inlet.axis("equal")
inlet.set_xbound(1.5,7.5)
inlet.set_ybound(3,8)
inlet.scatter(
    x=df_scambled["non-organelle"],
    y=df_scambled["cell-volume"],
    s=12,
    label="scambled",
    color="grey",alpha=0.5,edgecolors='w'
)
ax.set_adjustable('datalim')
x_min,x_max = dfs["non-organelle"].min(),dfs["non-organelle"].max()
y_min,y_max = dfs["cell-volume"].min(),dfs["cell-volume"].max()
ax.plot(
    [x_min,x_max],[y_min-0.5,y_min+(x_max-x_min)-0.5],"k--"
)
ax.plot(
    [x_min,x_max],[y_min+0.5,y_min+0.5+(2/3)*(x_max-x_min)],"r--"
)
ax.set_xlabel(r"$log(V_{cyto})$")
ax.set_ylabel(r"$log(V_{cell})$")
# ax.legend()
plt.savefig("plots/power_law/power_law_rectangular.png",dpi=600)
plt.close()



# generalize to other pairs of volumes.
def plot_loglog_vol(experiment,output=""):
    df_loglogs = df_bycell.loc[df_bycell["folder"].eq(experiments[experiment]) & df_bycell["organelle"].eq("ER"),["condition","cell-volume"]]
    df_loglogs = df_loglogs.reset_index().drop("index",axis=1)
    df_loglogs["cell-volume"] = np.log(df_loglogs["cell-volume"])
    for orga in [*organelles,"non-organelle"]:
        df_loglogs[orga] = np.log(df_bycell.loc[df_bycell["folder"].eq(experiments[experiment]) & df_bycell["organelle"].eq(orga),"total"].reset_index().drop("index",axis=1))
    df_loglogs.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_loglogs.dropna(axis=0,inplace=True)

    fitted = np.zeros((8,8))
    plt.rcParams['font.size'] = '26'
    fig_pair,pairplots = plt.subplots(
        nrows=8,ncols=8,
        figsize=(50,40),
        sharex=False,sharey=False
    )

    uniq_conds = np.sort(df_loglogs["condition"].unique())
    for i,prop1 in enumerate(["cell-volume",*organelles,"non-organelle"]): # row
        for j,prop2 in enumerate(["cell-volume",*organelles,"non-organelle"]): # col
            dfs2fit = []
            dfs2go  = []
            pct10x = np.percentile(df_loglogs[prop2],10)
            pct10y = np.percentile(df_loglogs[prop1],10)
            print(i,j,pct10y,pct10x)
            for condi in uniq_conds:
                dfs2fit.append(
                    df_loglogs[
                        df_loglogs["condition"].eq(condi) & 
                        df_loglogs[prop2].ge(pct10x) &
                        df_loglogs[prop1].ge(pct10y)
                ])
                dfs2go.append(
                    df_loglogs[
                        df_loglogs["condition"].eq(condi) & 
                        (df_loglogs[prop2].le(pct10x) |
                        df_loglogs[prop1].le(pct10y))
                ])
            dfs2fit = pd.concat(dfs2fit,ignore_index=True)
            dfs2go  = pd.concat(dfs2go, ignore_index=True)
            fitted[i,j] = LinearRegression().fit(dfs2fit[prop2].to_numpy().reshape(-1,1),dfs2fit[prop1]).coef_[0]
            # fitted[i,j] = scipy.odr.ODR(
            #         scipy.odr.Data(
            #             dfs2fit[prop2].to_numpy(),
            #             dfs2fit[prop1].to_numpy()
            #         ),
            #         scipy.odr.unilinear
            #     ).run().beta[0]
            for k,condi in enumerate(uniq_conds):
                pairplots[i,j].scatter(
                    dfs2fit.loc[dfs2fit["condition"].eq(condi),prop2],
                    dfs2fit.loc[dfs2fit["condition"].eq(condi),prop1],
                    color=sns.color_palette("tab10")[list_colors[experiment][k]],
                    label=f"{experiment} {condi}",
                    alpha=0.5,edgecolors='w'
                )
                pairplots[i,j].scatter(
                    dfs2go.loc[dfs2go["condition"].eq(condi),prop2],
                    dfs2go.loc[dfs2go["condition"].eq(condi),prop1],
                    color='k',alpha=0.3*(k+1)/len(uniq_conds),edgecolors='w'
                )
            xmin,xmax,xmean,ymean,k = dfs2go[prop2].min(),dfs2fit[prop2].max(),dfs2fit[prop2].mean(),dfs2fit[prop1].mean(),fitted[i,j]
            pairplots[i,j].plot(
                [xmin,xmax],[k*(xmin-xmean)+ymean,k*(xmax-xmean)+ymean],
                'k--'
            )
            # if i==7:
            #     pairplots[i,j].set_xlabel(f"log[V({prop2})]")
            # if j==0:
            #     pairplots[i,j].set_ylabel(f"log[V({prop1})]")
    pairplots[7,7].legend()
    if not Path(f"{output}/power_law/").exists():
        Path(f"{output}/power_law/").mkdir()
    fig_pair.savefig(f"{output}/power_law/pairwise_{experiment}.png")
    np.savetxt(f"{output}/power_law/pairwise_{experiment}.txt",fitted)
    print(f"Pairwise log-log regression: {experiment}")
    return None

for exps in exp_names:
    plot_loglog_vol(exps,output="data/REBUTTAL_PLOTS")

# plt.rcParams['font.size'] = '24'
# plt.figure()
# sns.pairplot(
#     data=dfs2fit,kind="reg",hue="condition",
#     vars=["cell-volume",*organelles,"non-organelle"],
#     palette=list(np.array(sns.color_palette("tab10"))[list_colors]),
#     plot_kws={"ci":None},diag_kws={"fill":False},
#     height=5.0
# )
# plt.savefig("data/power_law/pairwise_ge2.png")
# plt.close()


# generalize to other pairs of volumes, but only 100% glucose
df_glu_logs = df_bycell.loc[df_bycell["folder"].eq("EYrainbow_glucose_largerBF") & df_bycell["organelle"].eq("ER") &df_bycell["condition"].eq(100.),["cell-volume"]]
df_glu_logs = df_glu_logs.reset_index().drop("index",axis=1)
df_glu_logs["cell-volume"] = np.log(df_glu_logs["cell-volume"])
for orga in [*organelles,"non-organelle"]:
    df_glu_logs[orga] = np.log(df_bycell.loc[df_bycell["folder"].eq("EYrainbow_glucose_largerBF") & df_bycell["organelle"].eq(orga) & df_bycell["condition"].eq(100.),"total"].reset_index().drop("index",axis=1))
df_glu_logs.replace([np.inf, -np.inf], np.nan, inplace=True)
df_glu_logs.dropna(axis=0,inplace=True)

fitted = np.zeros((8,8))
for i,prop1 in enumerate(["cell-volume",*organelles,"non-organelle"]):
    for j,prop2 in enumerate(["cell-volume",*organelles,"non-organelle"]):
        x_min,x_max = np.nanpercentile(df_glu_logs.loc[:,prop1].to_numpy(),[10,99])
        # dfs2fit.append(df_glu_logs[df_glu_logs[prop1].ge(x_min)&df_glu_logs[prop1].le(x_max)])
        dfs2fit = df_glu_logs[df_glu_logs[prop1].ge(2)]
        # fitted[i,j] = np.polyfit(dfs2fit[prop1],dfs2fit[prop2],1)[0]
        fitted[i,j] = LinearRegression().fit(dfs2fit[prop1].to_numpy().reshape(-1,1),dfs2fit[prop2].to_numpy()).coef_[0]
np.savetxt("data/power_law/pairwise_glu-100.txt",fitted)

plt.figure()
sns.pairplot(
    data=dfs2fit,kind="reg",plot_kws={"ci":None},
    x_vars=["cell-volume",*organelles,"non-organelle"],
    y_vars=["cell-volume",*organelles,"non-organelle"]
)
plt.savefig("data/power_law/pairwise_glu-100.png")
plt.close


# Errorbar plot of V_cyto vs. growth rate
df_glu_cyto_rate = df_bycell.loc[df_bycell["folder"].eq("EYrainbow_glucose_largerBF")&df_bycell["organelle"].eq("non-organelle")]
df_glu_cyto_rate_bycondi = df_glu_cyto_rate[["condition","mean","total","cell-area","cell-volume","total-fraction","growth_rate"]].groupby("condition").mean()
df_glu_cyto_rate_bycondi["fraction-std"] = df_glu_cyto_rate[["condition","mean","total","cell-area","cell-volume","total-fraction","growth_rate"]].groupby("condition").std()["total-fraction"]
df_glu_cyto_rate_bycondi.reset_index(inplace=True)

plt.figure(figsize=(12,10))
for i,condi in enumerate(np.sort(df_glu_cyto_rate["condition"].unique())):
    plt.errorbar(
        df_glu_cyto_rate_bycondi.loc[df_glu_cyto_rate_bycondi["condition"].eq(condi),"growth_rate"],
        df_glu_cyto_rate_bycondi.loc[df_glu_cyto_rate_bycondi["condition"].eq(condi),"total-fraction"],
        yerr=df_glu_cyto_rate_bycondi.loc[df_glu_cyto_rate_bycondi["condition"].eq(condi),"fraction-std"]/np.sqrt(len(df_glu_cyto_rate_bycondi)),
        color=sns.color_palette("tab10")[list_colors["glucose"][i]],
        label=f"{condi/100*2}% glucose",
        fmt='o',capsize=10,markersize=20,alpha=0.5
    )
    plt.ylim(0.,0.7)
plt.xlabel("Growth Rate (hour$^{-1}$)")
plt.ylabel(r"Cytoplasmic Volume Fraction")
# ax_cyto_rate = plt.gca()
# handles, labels = ax_cyto_rate.get_legend_handles_labels()
# handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
# ax_cyto_rate.legend(handles, labels)
plt.tight_layout()
plt.savefig("data/REBUTTAL_PLOTS/power_law/cytofraction_vs_growthrate_sem.png")
plt.close()

# Errorbar plot of V_vacuole vs. growth rate
df_glu_vacu_rate = df_bycell.loc[df_bycell["folder"].eq("EYrainbow_glucose_largerBF")&df_bycell["organelle"].eq("vacuole")]
df_glu_vacu_rate_bycondi = df_glu_vacu_rate.groupby("condition").mean()
df_glu_vacu_rate_bycondi["fraction-std"] = df_glu_vacu_rate.groupby("condition").std()["total-fraction"]
df_glu_vacu_rate_bycondi.reset_index(inplace=True)

plt.figure(figsize=(12,10))
for i,condi in enumerate(np.sort(df_glu_vacu_rate["condition"].unique())):
    plt.errorbar(
        df_glu_vacu_rate_bycondi.loc[df_glu_vacu_rate_bycondi["condition"].eq(condi),"growth_rate"],
        df_glu_vacu_rate_bycondi.loc[df_glu_vacu_rate_bycondi["condition"].eq(condi),"total-fraction"],
        yerr=df_glu_vacu_rate_bycondi.loc[df_glu_vacu_rate_bycondi["condition"].eq(condi),"fraction-std"]/np.sqrt(len(df_glu_vacu_rate_bycondi)),
        color=sns.color_palette("tab10")[list_colors["glucose"][i]],
        label=f"{condi/100*2}% glucose",
        fmt='o',capsize=10,markersize=20,alpha=0.5
    )
    plt.ylim(0.,0.2)
plt.xlabel("Growth Rate (hour$^{-1}$)")
plt.ylabel("Vacuole Volume Fraction")
# ax_cyto_rate = plt.gca()
# handles, labels = ax_cyto_rate.get_legend_handles_labels()
# handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
# ax_cyto_rate.legend(handles, labels)
plt.tight_layout()
plt.savefig("data/power_law/vacuole_fraction_vs_growthrate_sem.png")
plt.close()


# Plot segmentation error
df_error = pd.read_csv("data/image_error_probability_new.csv")

summary_error = df_error.groupby(["organelle","experiment"]).mean()
summary_error.reset_index(inplace=True)

plt.figure(figsize=(20,12))
plt.rcParams['font.size'] = '26'
bar_positions = {}
for k,experiment in enumerate(exp_names):
    bar_positions = np.arange(6) + (k+1.5)/(len(exp_names)+2.)
    plt.bar(
        x=bar_positions,
        height=summary_error.loc[
            summary_error["experiment"].eq(experiments[experiment]),
            "error_upper"
        ].values[[4,5,0,2,3,1]]*100,
        width=1/(len(exp_names)+2.),
        label=experiment,
        color=sns.color_palette('tab10')[k],
    )
    print(summary_error.loc[
            summary_error["experiment"].eq(experiments[experiment]),
            "organelle"
        ].values[[4,5,0,2,3,1]])
for k,experiment in enumerate(exp_names):
    bar_positions = np.arange(6) + (k+1.5)/(len(exp_names)+2.)
    plt.bar(
        x=bar_positions,
        height= -summary_error.loc[
            summary_error["experiment"].eq(experiments[experiment]),
            "error_lower"
        ].values[[4,5,0,2,3,1]]*100,
        width=1/(len(exp_names)+2.),
        color=sns.color_palette('tab10')[k],
        alpha=0.5
    )
plt.axhline(color='k')
plt.xticks(np.arange(6)+0.5,organelles)
plt.xlabel("Organelle")
plt.ylabel("Segmentation Error / %")
plt.legend(loc=(1.04,0.5))
plt.savefig("data/image_error.png")

