# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
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
plt.rcParams['font.size'] = '16'
cmap = plt.get_cmap("tab10")
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

by_organlle = csv_mcmc2575[["organelle","std/mean","diff/mean"]].groupby("organelle").mean()

# %%
df_bycell = read_results(Path("./data"),subfolders,(px_x,px_y,px_z))

# %%
def make_pca_plots(experiment,property,groups=None,has_volume=False,is_normalized=True,non_organelle=False,saveto="./plots/"):
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
    
    if is_normalized:
        for col in columns:
            df_pca[col] = (df_pca[col]-df_pca[col].mean())/df_pca[col].std()
    
    df_pca.reset_index(inplace=True)

    if saveto is None:
        return df_pca

df_pca_raw = make_pca_plots(
    experiment="glucose",
    property="total-fraction",
    groups=None,
    has_volume=False,
    is_normalized=False, # be careful of this
    non_organelle=False,
    saveto=None
)

# %% TODO: extract this into a function in organelle_measure/
rng = np.random.default_rng()

N = 1000
centroids = np.zeros((N,6,6))     # (simulation, glucose condition  , organelle)
pc_components = np.zeros((N,6,6)) # (simulation, principal component, organelle)
for t in range(N):
    df_pca = df_pca_raw.copy()
    for o,organelle in enumerate(organelles):
        multiplier = rng.normal(loc=1.0,scale=by_organlle.loc[organelle,"std/mean"],size=len(df_pca_raw))
        df_pca.loc[:,organelle] = df_pca_raw.loc[:,organelle] * multiplier
        df_pca[organelle] = (df_pca[organelle]-df_pca[organelle].mean())/df_pca[organelle].std()

    df_centroid = df_pca.groupby("condition")[organelles].mean()

    fitter_centroid = LinearRegression(fit_intercept=False)
    np_centroid = df_centroid.to_numpy()
    fitter_centroid.fit(np_centroid,np.ones(np_centroid.shape[0]))

    vec_centroid_start = df_centroid.loc[extremes["EYrainbow_glucose_largerBF"][0],:].to_numpy()
    vec_centroid_start[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_start[:-1]))/fitter_centroid.coef_[-1]

    vec_centroid_ended = df_centroid.loc[extremes["EYrainbow_glucose_largerBF"][-1],:].to_numpy()
    vec_centroid_ended[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_ended[:-1]))/fitter_centroid.coef_[-1]

    vec_centroid = vec_centroid_ended - vec_centroid_start
    vec_centroid = vec_centroid/np.linalg.norm(vec_centroid)


    np_pca = df_pca[organelles].to_numpy()
    pca = PCA(n_components=6)
    pca.fit(np_pca)
    pca_components = pca.components_

    centroids[t] = np_centroid
    pc_components[t] = pca_components


# %% real data from experiment
df_pca = df_pca_raw.copy()
for o,organelle in enumerate(organelles):
    df_pca[organelle] = (df_pca[organelle]-df_pca[organelle].mean())/df_pca[organelle].std()

df_centroid = df_pca.groupby("condition")[organelles].mean()
fitter_centroid = LinearRegression(fit_intercept=False)
np_centroid = df_centroid.to_numpy()
fitter_centroid.fit(np_centroid,np.ones(np_centroid.shape[0]))

vec_centroid_start = df_centroid.loc[extremes["EYrainbow_glucose_largerBF"][0],:].to_numpy()
vec_centroid_start[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_start[:-1]))/fitter_centroid.coef_[-1]

vec_centroid_ended = df_centroid.loc[extremes["EYrainbow_glucose_largerBF"][-1],:].to_numpy()
vec_centroid_ended[-1] = (1 - np.dot(fitter_centroid.coef_[:-1],vec_centroid_ended[:-1]))/fitter_centroid.coef_[-1]

vec_centroid = vec_centroid_ended - vec_centroid_start
vec_centroid = vec_centroid/np.linalg.norm(vec_centroid)

np_pca = df_pca[organelles].to_numpy()
pca = PCA(n_components=6)
pca.fit(np_pca)
pca_components = pca.components_

# align pc with condition vectors
cosine_pca = np.dot(pca_components,vec_centroid)
for c in range(len(cosine_pca)):
    if cosine_pca[c] < 0:
        pca_components[c] = -pca_components[c]
cosine_pca = np.abs(cosine_pca)

arg_cosine = np.argsort(cosine_pca)[::-1]
pca_components_sorted = pca_components[arg_cosine]


# %%
fig3d = plt.figure(layout='compressed')
ax = fig3d.add_subplot(projection='3d')
for g in range(6): # g stands for glucose
    proj_centroids_first  = np.dot(centroids[:,g,:],pca_components[arg_cosine[0]])
    proj_centroids_second = np.dot(centroids[:,g,:],pca_components[arg_cosine[1]])
    proj_centroids_third  = np.dot(centroids[:,g,:],pca_components[arg_cosine[2]])

    ax.scatter(
        proj_centroids_first, proj_centroids_second, proj_centroids_third,
        s=10,linewidths=0.5,edgecolor='white',facecolor=cmap(list_colors["glucose"][g]),
        label=f"{df_centroid.index[g]*2/100} %"
    )
    ax.scatter(
        [np.dot(np_centroid[g],pca_components[arg_cosine[0]])],
        [np.dot(np_centroid[g],pca_components[arg_cosine[1]])],
        [np.dot(np_centroid[g],pca_components[arg_cosine[2]])],
        marker="X",s=100,edgecolor='white',facecolor=cmap(list_colors["glucose"][g]),
    )
ax.legend(loc=(1.15,0.5),fontsize=14)
ax.set_title("Centroid Projection onto PC")
ax.set_xlabel(f"Projection {arg_cosine[0]}",labelpad=5)
ax.set_ylabel(f"Projection {arg_cosine[1]}",labelpad=10)
ax.set_zlabel(f"Projection {arg_cosine[2]}",labelpad=10)
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
fig3d.savefig(
    "plots/pca_error_analysis/pca-error_glucose-1000_3d.png",
    dpi=600
)
plt.close(fig3d)

# %%
fig3d = plt.figure(layout='compressed')
ax = fig3d.add_subplot(projection='3d')
for g in [0,-2]: # g stands for glucose
    proj_centroids_first  = np.dot(centroids[:,g,:],pca_components[arg_cosine[0]])
    proj_centroids_second = np.dot(centroids[:,g,:],pca_components[arg_cosine[1]])
    proj_centroids_third  = np.dot(centroids[:,g,:],pca_components[arg_cosine[2]])

    ax.scatter(
        proj_centroids_first, proj_centroids_second, proj_centroids_third,
        s=10,linewidths=0.5,edgecolor='white',facecolor=cmap(list_colors["glucose"][g]),
        label=f"{df_centroid.index[g]*2/100} %"
    )
    ax.scatter(
        [np.dot(np_centroid[g],pca_components[arg_cosine[0]])],
        [np.dot(np_centroid[g],pca_components[arg_cosine[1]])],
        [np.dot(np_centroid[g],pca_components[arg_cosine[2]])],
        marker="X",s=100,edgecolor='white',facecolor=cmap(list_colors["glucose"][g]),
    )
ax.legend(loc=(1.15,0.5),fontsize=14)
ax.set_title("Centroid Projection onto PC")
ax.set_xlabel(f"Projection {arg_cosine[0]}",labelpad=5)
ax.set_ylabel(f"Projection {arg_cosine[1]}",labelpad=10)
ax.set_zlabel(f"Projection {arg_cosine[2]}",labelpad=10)
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
fig3d.savefig(
    "plots/pca_error_analysis/extremes_pca-error_glucose-1000_3d.png",
    dpi=600
)
plt.close(fig3d)

# %%
for st,nd in ((0,1),(0,2),(1,2)):
    plt.figure()
    for g in range(6):
        proj_centroids_1st = np.dot(centroids[:,g,:],pca_components[arg_cosine[st]])
        proj_centroids_2nd = np.dot(centroids[:,g,:],pca_components[arg_cosine[nd]])

        plt.scatter(
            proj_centroids_1st,proj_centroids_2nd,
            edgecolor='white',facecolor=cmap(list_colors["glucose"][g]),
            label=f"{df_centroid.index[g]*2/100} %"
        )
        plt.scatter(
            [np.dot(np_centroid[g],pca_components[arg_cosine[st]])],
            [np.dot(np_centroid[g],pca_components[arg_cosine[nd]])],
            marker="X",s=100,edgecolor='white',facecolor=cmap(list_colors["glucose"][g]),
        )
    plt.legend(fontsize=14)
    plt.xlabel(f"Projection {arg_cosine[st]}")
    plt.ylabel(f"Projection {arg_cosine[nd]}")
    plt.savefig(
        f"plots/pca_error_analysis/pca-error_glucose-1000_2d-{arg_cosine[st]}-{arg_cosine[nd]}.png",
        dpi=600
    )
    

# %% compare distribution of synthetic data with true centroids
for g0 in range(6): # g stands for glucose
    true_centroid = np_centroid[g0]
    subtract_same = centroids[:,g0,:] - true_centroid
    distance_same = np.apply_along_axis(np.linalg.norm,axis=1,arr=subtract_same)

    w,h = figaspect(0.7)
    plt.figure(figsize=(w,h))
    plt.hist(
        distance_same, density=True,
        color=cmap(list_colors["glucose"][g0]),histtype='step',
        label=f"{df_centroid.index[g0]*2/100}%"
    )
    for g1 in range(6):
        if g1==g0:
            continue
        # distance_centroid = np.linalg.norm((np_centroid[g1] - true_centroid))
        subtract_across = centroids[:,g1,:] - true_centroid
        distance_across = np.apply_along_axis(np.linalg.norm,axis=1,arr=subtract_across)
        plt.hist(
            distance_across,density=True,
            color=cmap(list_colors["glucose"][g1]),histtype='step',
            label=f"{df_centroid.index[g1]*2/100}%"
        )
        # plt.hist(
        #     [distance_centroid],
        #     bins=[distance_centroid-0.001,distance_centroid+0.001],
        #     density=True,
        #     color=cmap(list_colors["glucose"][g1]),histtype='stepfilled',
        # )
    plt.legend(loc=(1.05,0.4))
    plt.title(f"Distance from Centroid of {df_centroid.index[g0]*2/100}% Glucose")
    plt.xlabel(f"2-Norm Distance")

    # plt.show()
    plt.savefig(
        f"plots/pca_error_analysis/distance-distrib_glucose-{str(df_centroid.index[g0]).replace('.','-')}.png",
        dpi=600
    )

# %% compare distribution of synthetic data with true centroids
for g0 in [0,-2]: # g stands for glucose
    true_centroid = np_centroid[g0]
    subtract_same = centroids[:,g0,:] - true_centroid
    distance_same = np.apply_along_axis(np.linalg.norm,axis=1,arr=subtract_same)

    w,h = figaspect(0.7)
    plt.figure(figsize=(w,h))
    plt.hist(
        distance_same, density=True,
        color=cmap(list_colors["glucose"][g0]),histtype='step',
        label=f"{df_centroid.index[g0]*2/100}%"
    )
    for g1 in [0,-2]:
        if g1==g0:
            continue
        # distance_centroid = np.linalg.norm((np_centroid[g1] - true_centroid))
        subtract_across = centroids[:,g1,:] - true_centroid
        distance_across = np.apply_along_axis(np.linalg.norm,axis=1,arr=subtract_across)
        plt.hist(
            distance_across,density=True,
            color=cmap(list_colors["glucose"][g1]),histtype='step',
            label=f"{df_centroid.index[g1]*2/100}%"
        )
        # plt.hist(
        #     [distance_centroid],
        #     bins=[distance_centroid-0.001,distance_centroid+0.001],
        #     density=True,
        #     color=cmap(list_colors["glucose"][g1]),histtype='stepfilled',
        # )
    plt.legend(loc=(1.05,0.4))
    plt.title(f"Distance from Centroid of {df_centroid.index[g0]*2/100}% Glucose")
    plt.xlabel(f"2-Norm Distance")

    # plt.show()
    plt.savefig(
        f"plots/pca_error_analysis/extremes_distance-distrib_glucose-{str(df_centroid.index[g0]).replace('.','-')}.png",
        dpi=600
    )

# %%
