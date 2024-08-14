
# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,measure
from batch_apply import batch_apply

# %%
def parse_meta_organelle(name):
    """name is the stem of the ORGANELLE label image file."""
    organelle = name.partition("-")[2].partition("_")[0]
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
        "organelle":  organelle
    }

def measure1organelle(path_in,path_cell,path_out,metadata=None):
    print(f">>> finished {path_out.stem}.")
    return None

organelles = [
    "peroxisome",
    "ER",
    "golgi",
    "mitochondria",
    "LD",
    "vacuole"
]

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

folder_i = "./images/labelled"
folder_c = "./images/cell"
folder_o = "./data/results/"

print("Creating folders...")
for folder in subfolders:
    if (newfolder:=Path(folder_o)/folder).exists():
        print(f"Folder `{newfolder.name}` exists. Skip...")
    else:
        newfolder.mkdir()

list_i = []
list_c = []
list_o = []
for subfolder in subfolders:
    for path_c in (Path(folder_c)/subfolder).glob("*.tif"):
        for organelle in organelles:
            path_i = Path(folder_i)/subfolder/f"label-{organelle}_{path_c.stem.partition('_')[2]}.tiff"
            path_o = Path(folder_o)/subfolder/f"{organelle}_{path_c.stem.partition('_')[2]}.csv"

            list_i.append(path_i)
            list_c.append(path_c)
            list_o.append(path_o)
        
args = pd.DataFrame({
    "path_in":   list_i,
    "path_cell": list_c,
    "path_out":  list_o
})

batch_apply(measure1organelle,args)

# %% rebuttal at 8 hours
list_i = []
list_c = []
list_o = []

for path_c in Path("images/cell/paperRebuttal").glob("*.tif"):
    for organelle in organelles:
        path_i = Path("images/labelled/paperRebuttal")/f"label-{organelle}_{path_c.stem.partition('_')[2]}.tiff"
        path_o = Path("data/results/paperRebuttal")/f"{organelle}_{path_c.stem.partition('_')[2]}.csv"
        list_i.append(path_i)
        list_c.append(path_c)
        list_o.append(path_o)
        
args = pd.DataFrame({
    "path_in":   list_i,
    "path_cell": list_c,
    "path_out":  list_o
})
batch_apply(measure1organelle,args)


# %% rebuttal 2 labels on 1 organelle
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,measure
from batch_apply import batch_apply

def parse_meta_organelle(name):
    """name is the stem of the ORGANELLE label image file."""
    organelle = name.partition("_")[0].partition("-")[2]
    camera = name.partition("_")[2].partition("_")[0]
    field = name.partition("field-")[2]    
    return {
        "organelle":  organelle,
        "camera":     camera,
        "field":      field,
    }

def measure1organelle(path_in,path_cell,path_out,metadata=None):
    # parse metadata from filename
    name = Path(path_in).stem

    meta = parse_meta_organelle(name) if metadata is None else metadata

    img_orga = io.imread(str(path_in))
    img_cell = io.imread(str(path_cell))
    
    dfs = []
    for cell in measure.regionprops(img_cell):
        meta["idx-cell"] = cell.label
        min_row, min_col, max_row, max_col = cell.bbox
        img_orga_crop = img_orga[:,min_row:max_row,min_col:max_col]
        img_cell_crop = cell.image
        for z in range(img_orga_crop.shape[0]):
            img_orga_crop[z] = img_orga_crop[z]*img_cell_crop
        if not meta["organelle"] == "VO":
            measured_orga = measure.regionprops_table(
                img_orga_crop,
                properties=('label','area','bbox_area','bbox')
            )
        else:
            vacuole_area = 0
            vacuole_bbox_area = 0
            bbox0,bbox1,bbox2,bbox3,bbox4,bbox5 = 0,0,0,0,0,0
            for z in range(img_orga_crop.shape[0]):
                vacuole = measure.regionprops_table(
                    img_orga_crop[z],
                    properties=('label','area','bbox_area','bbox')
                )
                if len(vacuole["area"]) == 0:
                    continue
                if (maxblob:=max(vacuole["area"])) > vacuole_area:
                    vacuole_area = maxblob
                    idxblob = np.argmax(vacuole["area"])
                    vacuole_bbox_area = vacuole["bbox_area"][idxblob]
                    bbox0,bbox3 = z,z
                    bbox1,bbox2,bbox4,bbox5 = [vacuole[f"bbox-{i}"][idxblob] for i in range(4)]
            if vacuole_area==0:
                continue
            measured_orga = {
                'label': [0],
                'area':  [vacuole_area],
                "bbox_area": [vacuole_bbox_area],
                "bbox-0": [bbox0],
                "bbox-1": [bbox1],
                "bbox-2": [bbox2],
                "bbox-3": [bbox3],
                "bbox-4": [bbox4],
                "bbox-5": [bbox5],
            }
        result = meta | measured_orga
        dfs.append(pd.DataFrame(result))
    if len(dfs) == 0:
        print(f">>> {path_out} has no cells, skipped.")
        return None
    df_orga = pd.concat(dfs,ignore_index=True)
    df_orga.rename(columns={'label':'idx-orga',"area":"volume-pixel",'bbox_area':'volume-bbox'},inplace=True)
    df_orga.to_csv(str(path_out),index=False)
    print(f">>> finished {path_out.stem}.")
    return None

# %%
folder_c = "./images/cell/2024-06-25_2colorDiploidMeasure"
folder_i = "./images/labelled/2024-06-25_2colorDiploidMeasure"
folder_o = "./data/2024-06-25_2colorDiploidMeasure"

list_i = []
list_c = []
list_o = []
for path_c in (Path(folder_c)).glob("GL*.tif"):
    organelle = path_c.stem.partition("_"     )[0]
    field     = path_c.stem.partition("field-")[2]
    for path_i in Path(folder_i).glob(f"label-{organelle}*{field}.tif"):
        camera = path_i.stem.partition("_")[2].partition("_")[0]
        path_o = Path(folder_o)/f"{organelle}_field-{field}_{camera}.csv"
        list_i.append(path_i)
        list_c.append(path_c)
        list_o.append(path_o)

args = pd.DataFrame({
    "path_in":   list_i,
    "path_cell": list_c,
    "path_out":  list_o
})
# %%
batch_apply(measure1organelle,args)

# %%
