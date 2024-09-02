# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io,util,measure
from batch_apply import batch_apply

cmap = plt.get_cmap("tab10")

px_x,px_y,px_z = 0.41, 0.41, 0.20
organelles = [
    "peroxisome",
    "vacuole",
    "ER",
    "golgi",
    "mitochondria",
    "LD"
]
folder_img = Path("images/rebuttal_manual")
folder_data = Path("data/EYrainbow_glucose_largerBF/")

# %% find exactly where the sample image is cropped from
# img_full = io.imread("images/cell/EYrainbow_glucose_largerBF/binCell_EYrainbow_glu-100_field-0.tif")
# img_crop = io.imread(f"{folder_img}/binCell_EYrainbow_glu-100_field-0_crop.tif")

# for r in range(512-177):
#     for c in range(512-138):
#         if np.all(img_full[r:r+177,c:c+138]==img_crop):
#             print(r,c)
# # >>> 270 342

# %% white pixels in ImageJ uint16 images are not 'white'

# whites = {
#     "peroxisome":   1608,
#     "vacuole":        80,
#     "ER":            179,
#     "golgi":          12,
#     "mitochondria":  255,
#     "LD":            255,
# }

# for path_painted in folder_img.glob("painted*.tif"):
#     organelle = path_painted.stem.partition('_')[2].partition('_')[0]
    
#     painted = io.imread(str(path_painted))
#     corrected = (painted==whites[organelle])

#     io.imsave(
#         f"{folder_img}/equal_{path_painted.name}",
#         util.img_as_ubyte(corrected)
#     )

# %% [markdown] labeling of the images are done with the postprocessing script.

# %% measure organelles
# def parse_meta_organelle(name):
#     """name is the stem of the ORGANELLE label image file."""

#     return {
#         "experiment": "glucose",
#         "condition":  "100",
#         "hour":       3,
#         "field":      0,
#         "organelle":  name.partition("_")[2].partition("_")[0]
#     }


# def measure1organelle(path_in,path_cell,path_out,metadata=None):
#     # parse metadata from filename
#     name = Path(path_in).stem

#     meta = parse_meta_organelle(name) if metadata is None else metadata

#     img_orga = io.imread(str(path_in))
#     img_cell = io.imread(str(path_cell))
    
#     dfs = []
#     for cell in measure.regionprops(img_cell):
#         meta["idx-cell"] = cell.label
#         min_row, min_col, max_row, max_col = cell.bbox
#         img_orga_crop = img_orga[:,min_row:max_row,min_col:max_col]
#         img_cell_crop = cell.image
#         for z in range(img_orga_crop.shape[0]):
#             img_orga_crop[z] = img_orga_crop[z]*img_cell_crop
#         if not meta["organelle"] == "vacuole":
#             measured_orga = measure.regionprops_table(
#                 img_orga_crop,
#                 properties=('label','area','bbox_area','bbox')
#             )
#         else:
#             vacuole_area = 0
#             vacuole_bbox_area = 0
#             bbox0,bbox1,bbox2,bbox3,bbox4,bbox5 = 0,0,0,0,0,0
#             for z in range(img_orga_crop.shape[0]):
#                 vacuole = measure.regionprops_table(
#                     img_orga_crop[z],
#                     properties=('label','area','bbox_area','bbox')
#                 )
#                 if len(vacuole["area"]) == 0:
#                     continue
#                 if (maxblob:=max(vacuole["area"])) > vacuole_area:
#                     vacuole_area = maxblob
#                     idxblob = np.argmax(vacuole["area"])
#                     vacuole_bbox_area = vacuole["bbox_area"][idxblob]
#                     bbox0,bbox3 = z,z
#                     bbox1,bbox2,bbox4,bbox5 = [vacuole[f"bbox-{i}"][idxblob] for i in range(4)]
#             if vacuole_area==0:
#                 continue
#             measured_orga = {
#                 'label': [0],
#                 'area':  [vacuole_area],
#                 "bbox_area": [vacuole_bbox_area],
#                 "bbox-0": [bbox0],
#                 "bbox-1": [bbox1],
#                 "bbox-2": [bbox2],
#                 "bbox-3": [bbox3],
#                 "bbox-4": [bbox4],
#                 "bbox-5": [bbox5],
#             }
#         result = meta | measured_orga
#         dfs.append(pd.DataFrame(result))
#     if len(dfs) == 0:
#         print(f">>> {path_out} has no cells, skipped.")
#         return None
#     df_orga = pd.concat(dfs,ignore_index=True)
#     df_orga.rename(columns={'label':'idx-orga',"area":"volume-pixel",'bbox_area':'volume-bbox'},inplace=True)
#     df_orga.to_csv(str(path_out),index=False)
#     print(f">>> finished {path_out.stem}.")
#     return None

# # %% 
# list_in   = []
# list_cell = []
# list_out  = []
# for organelle in organelles:
#     list_in.append(f"{folder_img}/label_{organelle}_EYrainbow_glu-100_field-0_crop.tif")
#     list_cell.append(f"{folder_img}/binCell_EYrainbow_glu-100_field-0_crop.tif")
#     list_out.append(f"{folder_img}/{organelle}_EYrainbow_glu-100_field-0_crop.csv")
# args = pd.DataFrame({
#     "path_in":   list_in,
#     "path_cell": list_cell,
#     "path_out":  list_out,
# })

# # %%
# batch_apply(measure1organelle,args)


# # %% try directly measure bw images instread of label images, 
# # because we only need total volume
# def parse_meta_organelle(name):
#     """name is the stem of the ORGANELLE label image file."""

#     return {
#         "experiment": "glucose",
#         "condition":  "100",
#         "hour":       3,
#         "field":      0,
#         "organelle":  name.partition("_")[2].partition("_")[2].partition("_")[0]
#     }

# list_in   = []
# list_cell = []
# list_out  = []
# for organelle in organelles:
#     list_in.append(f"{folder_img}/equal_painted_{organelle}_EYrainbow_glu-100_field-0_crop.tif")
#     list_cell.append(f"{folder_img}/binCell_EYrainbow_glu-100_field-0_crop.tif")
#     list_out.append(f"{folder_img}/bw_{organelle}_EYrainbow_glu-100_field-0_crop.csv")
# args = pd.DataFrame({
#     "path_in":   list_in,
#     "path_cell": list_cell,
#     "path_out":  list_out,
# })
# # %%
# batch_apply(measure1organelle,args)



# %% access data of cells and organelles 
img_cell = io.imread(str(folder_img/"binCell_EYrainbow_glu-100_field-0_crop.tif"))
cell_indices = np.unique((img_cell[img_cell>0]))

csv_cell = pd.read_csv(f"{folder_data}/cell_EYrainbow_glu-100_field-0.csv")

# Cell
dfs_cell = []
for c in cell_indices:
    dfs_cell.append(csv_cell.loc[csv_cell['idx-cell'].eq(c),:])
df_cell = pd.concat(dfs_cell,ignore_index=True)


# Organelle - paper
dfs_orga = []
for organelle in organelles:
    csv_orga = pd.read_csv(f"{folder_data}/{organelle}_EYrainbow_glu-100_field-0.csv")
    for c in cell_indices:
        dfs_orga.append(csv_orga.loc[csv_orga['idx-cell'].eq(c),:])
df_orga = pd.concat(dfs_orga,ignore_index=True)


# Organelle - manual segmentation
df_manual = pd.concat((pd.read_csv(str(fmanual)) for fmanual in folder_img.glob("*.csv")))


# %% Data procssing; cancelled um volume calculation on 2024-08-23

# df_cell.loc[:,"effective-volume"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*df_cell.loc[:,"area"]*np.sqrt(df_cell.loc[:,"area"])/np.sqrt(np.pi) 
pivot_cell_bycell = df_cell.set_index(["idx-cell"])

# df_orga["volume-micron"] = np.empty_like(df_orga.index)
# df_orga.loc[df_orga["organelle"].eq("vacuole"),"volume-micron"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*df_orga.loc[df_orga["organelle"].eq("vacuole"),"volume-pixel"]*np.sqrt(df_orga.loc[df_orga["organelle"].eq("vacuole"),"volume-pixel"])/np.sqrt(np.pi) 
# df_orga.loc[df_orga["organelle"].ne("vacuole"),"volume-micron"] = px_x*px_y*px_z*df_orga.loc[df_orga["organelle"].ne("vacuole"),"volume-pixel"]
# pivot_orga_bycell_mean = df_orga.loc[:,["organelle","idx-cell",'method',"volume-micron"]].groupby(["organelle","idx-cell",'method']).mean( )["volume-micron"]
# pivot_orga_bycell_nums = df_orga.loc[:,["organelle","idx-cell",'method',"volume-micron"]].groupby(["organelle","idx-cell",'method']).count()["volume-micron"]
# pivot_orga_bycell_totl = df_orga.loc[:,["organelle","idx-cell",'method',"volume-micron"]].groupby(["organelle","idx-cell",'method']).sum(  )["volume-micron"]
df_orga["method"] = 'ilastik'
pivot_orga_bycell_mean = df_orga.loc[:,["organelle","idx-cell",'method',"volume-pixel"]].groupby(["organelle","idx-cell",'method']).mean( )["volume-pixel"]
pivot_orga_bycell_nums = df_orga.loc[:,["organelle","idx-cell",'method',"volume-pixel"]].groupby(["organelle","idx-cell",'method']).count()["volume-pixel"]
pivot_orga_bycell_totl = df_orga.loc[:,["organelle","idx-cell",'method',"volume-pixel"]].groupby(["organelle","idx-cell",'method']).sum(  )["volume-pixel"]

# df_manual["volume-micron"] = np.empty_like(df_manual.index)
# df_manual.loc[df_manual["organelle"].eq("vacuole"),"volume-micron"] = (px_x*px_y)*np.sqrt(px_x*px_y)*(2.)*df_manual.loc[df_manual["organelle"].eq("vacuole"),"volume-pixel"]*np.sqrt(df_manual.loc[df_manual["organelle"].eq("vacuole"),"volume-pixel"])/np.sqrt(np.pi) 
# df_manual.loc[df_manual["organelle"].ne("vacuole"),"volume-micron"] = px_x*px_y*px_z*df_manual.loc[df_manual["organelle"].ne("vacuole"),"volume-pixel"]
# pivot_manual_bycell_mean = df_manual.loc[:,["organelle","idx-cell",'method',"volume-micron"]].groupby(["organelle","idx-cell",'method']).mean( )["volume-micron"]
# pivot_manual_bycell_nums = df_manual.loc[:,["organelle","idx-cell",'method',"volume-micron"]].groupby(["organelle","idx-cell",'method']).count()["volume-micron"]
# pivot_manual_bycell_totl = df_manual.loc[:,["organelle","idx-cell",'method',"volume-micron"]].groupby(["organelle","idx-cell",'method']).sum(  )["volume-micron"]
df_manual['method'] = 'manual'
pivot_manual_bycell_mean = df_manual.loc[:,["organelle","idx-cell",'method',"volume-pixel"]].groupby(["organelle","idx-cell",'method']).mean( )["volume-pixel"]
pivot_manual_bycell_nums = df_manual.loc[:,["organelle","idx-cell",'method',"volume-pixel"]].groupby(["organelle","idx-cell",'method']).count()["volume-pixel"]
pivot_manual_bycell_totl = df_manual.loc[:,["organelle","idx-cell",'method',"volume-pixel"]].groupby(["organelle","idx-cell",'method']).sum(  )["volume-pixel"]

# %%
index_bycell = pd.MultiIndex.from_tuples(
    [
        (organelle,index,id_method) 
        for organelle in organelles
        for index in pivot_cell_bycell.index.to_list() 
        for id_method in ["ilastik","manual"]
    ],
    names=["organelle","idx-cell","method"]
)

pivot_orga_bycell = pd.DataFrame(index=index_bycell)


pivot_orga_bycell.loc[pivot_orga_bycell_mean.index,'mean']  = pivot_orga_bycell_mean
pivot_orga_bycell.loc[pivot_orga_bycell_nums.index,'count'] = pivot_orga_bycell_nums
pivot_orga_bycell.loc[pivot_orga_bycell_totl.index,'total'] = pivot_orga_bycell_totl

pivot_orga_bycell.loc[pivot_manual_bycell_mean.index,'mean']  = pivot_manual_bycell_mean
pivot_orga_bycell.loc[pivot_manual_bycell_nums.index,'count'] = pivot_manual_bycell_nums
pivot_orga_bycell.loc[pivot_manual_bycell_totl.index,'total'] = pivot_manual_bycell_totl

df_orga_bycell = pivot_orga_bycell.reset_index()

for organelle in organelles:
    df_orga_bycell.loc[
        df_orga_bycell["organelle"].eq(organelle)
      & df_orga_bycell["method"].eq("ilastik"),
        "total_normed_max"
    ] = df_orga_bycell.loc[
            df_orga_bycell["organelle"].eq(organelle)
          & df_orga_bycell["method"].eq("ilastik"),
            "total"
    ] / df_orga_bycell.loc[
        df_orga_bycell["organelle"].eq(organelle)
      & df_orga_bycell["method"].eq("ilastik"),
        "total"
    ].max()
    df_orga_bycell.loc[
        df_orga_bycell["organelle"].eq(organelle)
      & df_orga_bycell["method"].eq("manual"),
        "total_normed_max"
    ] = df_orga_bycell.loc[
            df_orga_bycell["organelle"].eq(organelle)
          & df_orga_bycell["method"].eq("manual"),
            "total"
    ] / df_orga_bycell.loc[
        df_orga_bycell["organelle"].eq(organelle)
      & df_orga_bycell["method"].eq("manual"),
        "total"
    ].max()

# %%
names = {
    "peroxisome":   "Peroxisome",
    "vacuole":      "Vacuole",
    "ER":           "ER",
    "golgi":        "Golgi",
    "mitochondria": "Mitochondrion",
    "LD":           "Lipid Droplet",
}
correlations = []
differences = {}
for organelle in organelles:
    x = df_orga_bycell.loc[
            df_orga_bycell["organelle"].eq(organelle)
          & df_orga_bycell["method"].eq('ilastik'),
            ["idx-cell","total_normed_max"]
        ]

    y = df_orga_bycell.loc[
            df_orga_bycell["organelle"].eq(organelle)
          & df_orga_bycell["method"].eq('manual'),
            ['idx-cell',"total_normed_max"]
        ]

    x.set_index('idx-cell',inplace=True)
    y.set_index('idx-cell',inplace=True)
    xy = x.join(y,how='outer',lsuffix="_x",rsuffix="_y")
    # xy_zero = xy.copy()
    # xy_zero.loc[xy_zero["total_x"].isna(),"total_x"] = 0
    # xy_zero.loc[xy_zero["total_y"].isna(),"total_y"] = 0
    xy_drop = xy.dropna()
    
    corr = np.corrcoef(xy_drop,rowvar=False)
    correlations.append(corr[1,0])
    
    normed_diff = (xy_drop.loc[:,"total_normed_max_x"] - xy_drop.loc[:,"total_normed_max_y"])/xy_drop.loc[:,"total_normed_max_y"]
    differences[organelle] = normed_diff

    # plt.figure()
    # plt.title(f"{names[organelle]} Normalized Segmented Total Volume per Cell",fontsize=18)
    # plt.xlabel("Ilastik",fontsize=18)
    # plt.ylabel("Manual Segmentation",fontsize=18)
    # plt.scatter(x,y)
    # # plt.plot(
    # #     [x.min(),x.max()],
    # #     [x.min(),x.max()]
    # # )

    # # plt.show()
    # plt.savefig(f"plots/manual_segment/px_normalzied-max_scatter_{organelle}.png")

# %%
plt.figure()
plt.bar(np.arange(6),correlations)
plt.xticks(np.arange(6),organelles)
plt.xlabel("Organelle",fontsize=18)
plt.ylabel("Correlation Coefficient",fontsize=18)
# plt.show()
plt.savefig(f"plots/manual_segment/px_normalzied-max_correlation_coefficient.png")

# %%
fig, ax = plt.subplots()
plt.ylim((-1,1))
for o,organelle in enumerate(organelles):
    ax.scatter(
        o + 0.75 + 0.5*np.random.random(size=len(differences[organelle])),
        differences[organelle],
        s=2,c=cmap(o)
    )
bplot = ax.boxplot(
    [differences[o] for o in organelles],
    sym='',
    patch_artist = True,
    labels = organelles,
    medianprops=dict(color='black')
)
for p,patch in enumerate(bplot['boxes']):
    patch.set_facecolor(cmap(p))
# plt.show()
plt.savefig("plots/manual_segment/px_normalzied-max_error-percentage.png")
# %%
df_diff = pd.DataFrame({
	"organelle": organelles,
	"mean":   [differences[o].mean() for o in organelles],
	"median": [differences[o].median() for o in organelles],
})
df_diff.to_csv("plots/manual_segment/error-percentage.csv")
# %%
plt.figure()
plt.bar(
    np.arange(6),df_diff["mean"],
    color='white',edgecolor='black'
)
plt.xticks(np.arange(6),df_diff['organelle'],fontsize=10)
plt.xlabel("Organelle")
plt.ylabel("Mean Fractional Difference")
plt.savefig("plots/manual_segment/error_percentage_mean.png",dpi=600)
plt.close()


# %%
plt.figure()
plt.bar(
    np.arange(6),df_diff["median"],
    color='white',edgecolor='black'
)
plt.xticks(np.arange(6),df_diff['organelle'],fontsize=10)
plt.xlabel("Organelle")
plt.ylabel("median Fractional Difference")
plt.savefig("plots/manual_segment/error_percentage_median.png",dpi=600)
plt.close()

# %%
