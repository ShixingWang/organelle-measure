# %%
import pandas as pd
from pathlib import Path
from skimage import io,measure
from organelle_measure.tools import batch_apply

# %%
def parse_meta_cell(name):
    """name is the stem of the ORGANELLE label image file."""
    if "1nmpp1" in name:
        experiment = "1nmpp1"
    elif "betaEstrodiol" in name:
        experiment = "betaEstrodiol"
    else:
        experiment = name.partition("EYrainbow_")[2].partition("-")[0]
    condition = name.partition(f"{experiment}-")[2].partition("_")[0]
    field = name.partition("field-")[2]    
    return {
        "experiment": experiment,
        "condition":  condition,
        "hour":       3,
        "field":      field,
    }

def measure1cell(path_in,path_out):
    img_cell = io.imread(str(path_in))
    name = Path(path_in).stem
    meta = parse_meta_cell(name)
    measured = measure.regionprops_table(
                    img_cell,
                    properties=('label','area','centroid','bbox','eccentricity')
               )
    result = meta | measured
    df = pd.DataFrame(result)
    df.rename(columns={'label':'idx-cell'},inplace=True)
    df.to_csv(str(path_out),index=False)
    return None

# %%
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

folder_i = "./images/cell"
folder_o = "./data/results/"

print("Creating folders...")
for folder in subfolders:
    if (newfolder:=Path(folder_o)/folder).exists():
        print(f"Folder `{newfolder.name}` exists. Skip...")
    else:
        newfolder.mkdir()

list_i = []
list_o = []
for subfolder in subfolders:
    for path_i in (Path(folder_i)/subfolder).glob("*.tif"):
        path_o = (Path(folder_o)/subfolder)/f"cell_{path_i.stem.partition('_')[2]}.csv"
        list_i.append(path_i)
        list_o.append(path_o)

args = pd.DataFrame({
    "path_in":   list_i,
    "path_out":  list_o
})

batch_apply(measure1cell,args)

# %%
list_i = []
list_o = []

for path_i in Path("images/cell/paperRebuttal").glob("*.tif"):
    path_o = Path("data/results/paperRebuttal")/f"cell_{path_i.stem.partition('_')[2]}.csv"
    list_i.append(path_i)
    list_o.append(path_o)

args = pd.DataFrame({
    "path_in":   list_i,
    "path_out":  list_o
})
batch_apply(measure1cell,args)

# %%
csv = pd.read_csv("./binCell-after_EYrainbow_glu-0-5_field-0.csv")
# %%
import numpy as np
import matplotlib.pyplot as plt

px_x,px_y,px_z = 0.41,0.41,0.20
csv["effective-volume"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*csv.loc[:,"area"]*np.sqrt(csv.loc[:,"area"])/np.sqrt(np.pi)

plt.figure()
plt.hist(csv["area"],bins=20)
plt.xlabel("size/pixels")
plt.savefig("binCell-after_EYrainbow_glu-0-5_field-0_histogram-pixel.png")

plt.figure()
plt.hist(csv["effective-volume"],bins=20)
plt.xlabel("size/microns")
plt.savefig("binCell-after_EYrainbow_glu-0-5_field-0_histogram-microns.png")


# %% for 2-label-on-1-organelle strains
import pandas as pd
from pathlib import Path
from skimage import io,measure
from batch_apply import batch_apply

def parse_meta_cell(name):
    """name is the stem of the ORGANELLE label image file."""
    organelle = name.partition(f"_")[0]
    field = name.partition("field-")[2]    
    return {
        "organelle":  organelle,
        "field":      field,
    }

def measure1cell(path_in,path_out):
    img_cell = io.imread(str(path_in))
    name = Path(path_in).stem
    meta = parse_meta_cell(name)
    measured = measure.regionprops_table(
                    img_cell,
                    properties=('label','area','centroid','bbox','eccentricity')
               )
    result = meta | measured
    df = pd.DataFrame(result)
    df.rename(columns={'label':'idx-cell'},inplace=True)
    df.to_csv(str(path_out),index=False)
    return None

# %%
list_i = []
list_o = []

for path_i in Path("images/cell/2024-06-25_2colorDiploidMeasure").glob("*.tif"):
    stem = path_i.stem
    path_o = Path("data/2024-06-25_2colorDiploidMeasure")/f"cell_{stem.partition('_')[0]}_field-{stem.partition('field-')[2]}.csv"
    list_i.append(path_i)
    list_o.append(path_o)

args = pd.DataFrame({
    "path_in":   list_i,
    "path_out":  list_o
})
# %%
batch_apply(measure1cell,args)

# %%
