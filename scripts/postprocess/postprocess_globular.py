# %%
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,segmentation
from organelle_measure.tools import batch_apply

# %%
def postprocess_globular(path_in,path_ref,path_out):
    with h5py.File(str(path_in),'r') as f_in:
        img_in = f_in["exported_data"][:]
    img_in = (img_in[1]>img_in[0])
    img_ref = io.imread(str(path_ref))
    
    # idx_max = feature.peak_local_max(img_ref,min_distance=1)
    # img_max = np.zeros_like(img_ref,dtype=bool)
    # img_max[tuple(idx_max.T)] = True

    img_out = segmentation.watershed(-img_ref,mask=img_in)
    io.imsave(
        str(path_out),
        util.img_as_uint(img_out)
    )
    return None

# %%
folders = [
    # "EYrainbow_glucose",
    # "EYrainbow_glucose_largerBF",
    # "EYrainbow_rapamycin_1stTry",
    # "EYrainbow_rapamycin_CheckBistability",
    "EYrainbow_1nmpp1_1st" # ,
    # "EYrainbowWhi5Up_betaEstrodiol",
    # "EYrainbow_leucine_large",
    # "EYrainbow_leucine"
]
organelles = ["peroxisome","LD","golgi"]
folder_i = "./images/preprocessed/"
fodler_o = "./images/labelled/"


list_i   = []
list_ref = []
list_o   = []
for folder in folders:
    for organelle in organelles:
        for path_binary in (Path(folder_i)/folder).glob(f"probability_{organelle}*.h5"):
            path_output = (Path(fodler_o)/folder)/f"label-{path_binary.stem.partition('_')[2]}.tiff"
            path_ref = (Path(folder_i)/folder)/f"{path_binary.stem.partition('_')[2]}.tif"
            list_i.append(path_binary)
            list_ref.append(path_ref)
            list_o.append(path_output)
args = pd.DataFrame({
    "path_in":  list_i,
    "path_ref": list_ref,
    "path_out": list_o
})

batch_apply(postprocess_globular,args)

# %%
organelles = ["peroxisome","LD","golgi"]

list_i   = []
list_ref = []
list_o   = []
for organelle in organelles:
    for path_binary in Path("images/preprocessed/paperRebuttal").glob(f"probability_{organelle}*.h5"):
        path_output = Path("images/labelled/paperRebuttal")/f"label-{path_binary.stem.partition('_')[2]}.tiff"
        path_ref = Path("images/preprocessed/paperRebuttal")/f"{path_binary.stem.partition('_')[2]}.tif"
        list_i.append(path_binary)
        list_ref.append(path_ref)
        list_o.append(path_output)
args = pd.DataFrame({
    "path_in":  list_i,
    "path_ref": list_ref,
    "path_out": list_o
})
batch_apply(postprocess_globular,args)

# %% rebuttal dual labelled organlles
def postprocess_globular(path_in,path_ref,path_out):
    img_in  = io.imread(str(path_in))
    img_in  = (img_in>0.5)

    img_ref = io.imread(str(path_ref))
    
    img_out = segmentation.watershed(-img_ref,mask=img_in)
    io.imsave(
        str(path_out),
        util.img_as_uint(img_out)
    )
    return None

organelles = ["PX","LD","GL"]

list_i   = []
list_ref = []
list_o   = []
for organelle in organelles:
    for path_binary in Path("images/preprocessed/2024-06-25_2colorDiploidMeasure").glob(f"Probabilities_{organelle}*.tif"):
        path_output = Path("images/labelled/2024-06-25_2colorDiploidMeasure")/f"label-{path_binary.stem.partition('_')[2]}.tif"
        path_ref = Path("images/preprocessed/2024-06-25_2colorDiploidMeasure")/f"{path_binary.stem.partition('_')[2]}.tif"
        list_i.append(path_binary)
        list_ref.append(path_ref)
        list_o.append(path_output)
args = pd.DataFrame({
    "path_in":  list_i,
    "path_ref": list_ref,
    "path_out": list_o
})
# %%
batch_apply(postprocess_globular,args)

# %%
