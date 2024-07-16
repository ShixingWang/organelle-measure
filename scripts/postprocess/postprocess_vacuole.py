# %%
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,morphology,measure
from organelle_measure.tools import skeletonize_zbyz,watershed_zbyz,find_complete_rings,better_vacuole_img,batch_apply

# %%
def postprocess_vacuole(path_in,path_cell,path_out):
    with h5py.File(str(path_in),'r') as f_in:
        img_orga = f_in["exported_data"][:]
    img_orga = np.argmax(img_orga,axis=0)
    img_orga = (img_orga>0)

    img_cell = io.imread(str(path_cell))

    img_skeleton  = skeletonize_zbyz(img_orga)

    img_core      = find_complete_rings(img_skeleton)
    
    # img_vacuole   = better_vacuole_img(img_core,img_watershed)
    img_vacuole = np.zeros_like(img_core,dtype=int)
    for z in range(img_vacuole.shape[0]):
        sample = img_core[z]
        candidates = np.unique(sample[img_cell>0])
        for color in candidates:
            if len(np.unique(img_cell[sample==color]))==1:
                img_vacuole[z,sample==color] = color

    io.imsave(
        str(path_out),
        util.img_as_uint(img_vacuole) 
    )
    return None

# %%
folders = [
    "EYrainbow_glucose",
    "EYrainbow_glucose_largerBF",
    "EYrainbow_rapamycin_1stTry",
    "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st",
    "EYrainbowWhi5Up_betaEstrodiol",
    "EYrainbow_leucine_large", # "leucine-large-blue-gaussian"
    "EYrainbow_leucine"
]
folder_i = "./images/preprocessed/"
folder_c = "./images/cell/"
folder_o = "./images/labelled/"

print("Creating folders...")
for folder in folders:
    if (newfolder:=Path(folder_o)/folder).exists():
        print(f"Folder `{newfolder.name}` exists. Skip...")
    else:
        newfolder.mkdir()
        print(f"Folder `{newfolder.name}` created. Next...")

list_i = []
list_c = []
list_o = []
for folder in folders:
    for path_cell in (Path(folder_c)/folder).glob(f"*.tif"):
        if folder=="EYrainbow_leucine_large":
            path_binary = (Path(folder_i)/"leucine-large-blue-gaussian")/f"probability_spectral-blue_{path_cell.stem.partition('_')[2]}.h5"
        else:
            path_binary = (Path(folder_i)/folder)/f"probability_vacuole_{path_cell.stem.partition('_')[2]}.h5"
        path_output = (Path(folder_o)/folder)/f"label-vacuole_{path_cell.stem.partition('_')[2]}.tiff"
        list_i.append(path_binary)
        list_c.append(path_cell)
        list_o.append(path_output)

args = pd.DataFrame({
    "path_in":   list_i,
    "path_cell": list_c,
    "path_out":  list_o
})
# args.to_csv("./vauocle.csv",index=False)

batch_apply(postprocess_vacuole,args)

# %% paper rebuttal with cells after 8 hours

list_i = []
list_c = []
list_o = []
for path_cell in Path("images/cell/paperRebuttal").glob(f"*.tif"):
    path_binary = Path("images/preprocessed/paperRebuttal")/f"probability_vacuole_{path_cell.stem.partition('_')[2]}.h5"
    path_output = Path("images/labelled/paperRebuttal")/f"label-vacuole_{path_cell.stem.partition('_')[2]}.tiff"
    list_i.append(path_binary)
    list_c.append(path_cell)
    list_o.append(path_output)

args = pd.DataFrame({
    "path_in":   list_i,
    "path_cell": list_c,
    "path_out":  list_o
})
batch_apply(postprocess_vacuole,args)


# %%
def postprocess_vacuole(path_in,path_cell,path_out):
    img_orga = io.imread(str(path_in))
    img_orga = (img_orga>0.5)

    img_cell = io.imread(str(path_cell))

    img_skeleton  = skeletonize_zbyz(img_orga)

    img_core      = find_complete_rings(img_skeleton)
    
    # img_vacuole   = better_vacuole_img(img_core,img_watershed)
    img_vacuole = np.zeros_like(img_core,dtype=int)
    for z in range(img_vacuole.shape[0]):
        sample = img_core[z]
        candidates = np.unique(sample[img_cell>0])
        for color in candidates:
            if len(np.unique(img_cell[sample==color]))==1:
                img_vacuole[z,sample==color] = color

    io.imsave(
        str(path_out),
        util.img_as_uint(img_vacuole) 
    )
    return None

list_i = []
list_c = []
list_o = []
for path_binary in Path("images/preprocessed/2024-06-25_2colorDiploidMeasure").glob(f"Probabilities_VO*.tif"):
    path_cell = Path("images/cell/2024-06-25_2colorDiploidMeasure")/f"VO_binCell-after_{path_binary.stem.rpartition('_')[2]}.tif"
    path_output = Path("images/labelled/2024-06-25_2colorDiploidMeasure")/f"label-{path_binary.stem.partition('_')[2]}.tif"
    list_i.append(path_binary)
    list_c.append(path_cell)
    list_o.append(path_output)

args = pd.DataFrame({
    "path_in":   list_i,
    "path_cell": list_c,
    "path_out":  list_o
})
# %%
batch_apply(postprocess_vacuole,args)
# %%
