# This part is not used, mainly because this is not exactly 
# what the reviewer asked us to do.
import numpy as np
import pandas as pd
from pathlib import Path
from nd2reader import ND2Reader
from skimage import io,util,segmentation,measure,filters
from organelle_measure.yeaz import yeaz_preprocesses,yeaz_label
from organelle_measure.tools import skeletonize_zbyz,neighbor_mean,find_complete_rings


FOLDER = "2024-02-06_paperRebuttal3Colors"

for parent in [
	"images/raw",
	"images/cell",
	"images/preprocessed",
	"images/labelled",
	"data/results"
]:
	long = Path(parent)/FOLDER
	if not long.exists():
		Path.mkdir(long)

# Segment Cells
for path_cell in Path(f"images/raw/{FOLDER}").glob("camera*.nd2"):
	with ND2Reader(str(path_cell)) as nd2:
		nd2.bundle_axes = "yx"
		nd2.iter_axes = 't'
		img = nd2[0]
	for prep in yeaz_preprocesses:
		img = prep(img)
	segmented = yeaz_label(img, min_dist=5)
	segmented = segmentation.clear_border(segmented)
	properties = measure.regionprops(segmented)
	for prop in properties:
		if prop.area < 50:
			segmented[segmented==prop.label] = 0
	segmented = measure.label(segmented)
	output = np.zeros((512,512),dtype=int)
	shape0,shape1 = segmented.shape
	output[:shape0,:shape1] = segmented

	io.imsave(
		f"images/cell/{FOLDER}/{path_cell.stem}.tif",
		util.img_as_uint(output)
	)

# Preprocess
# peroxisome
path = Path(f"images/raw/{FOLDER}/z-37_unmixed-blue_EY2795WT-EY2796triColor_check-2.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes   = "c"
	image = nd2[0]
gauss = filters.gaussian(image,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/{FOLDER}/preprocessed_peroxisome_diploids3colors.tif",
	util.img_as_uint(gauss)
)
# vacuole
path = Path(f"images/raw/{FOLDER}/z-37_unmixed-blue_EY2795triColor-EY2796WT_check-2.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes   = "c"
	image = nd2[1]
gauss = filters.gaussian(image,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/{FOLDER}/preprocessed_vacuole_diploids3colors.tif",
	util.img_as_uint(gauss)
)
# ER
path = Path(f"images/raw/{FOLDER}/z-37_spectra-green_EY2795WT-EY2796triColor_check-2.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = 'zyx'
	nd2.iter_axes = "t"
	image = nd2[0]
gauss = filters.gaussian(image,sigma=0.3,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/{FOLDER}/preprocessed_ER_diploids3colors.tif",
	util.img_as_uint(gauss)
)
# golgi
path_cell = Path(f"images/cell/{FOLDER}/camera-BF-after_EY2795triColor-EY2796WT_check-2.tif")
cell = io.imread(str(path_cell))
path = Path(f"images/raw/{FOLDER}/z-37_spectra-yellow_EY2795triColor-EY2796WT_check-2.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "czyx"
	nd2.iter_axes = 't'
	image = nd2[0]
image = np.sum(image,axis=0)
bkgd = neighbor_mean(image,cell)
clear = image - bkgd
clear[clear<0] = 0
gauss = filters.gaussian(clear,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/{FOLDER}/preprocessed_golgi_diploids3colors.tif",
	util.img_as_uint(gauss)
)
# mitochondria
path_cell = Path(f"images/cell/{FOLDER}/camera-BF-after_EY2795WT-EY2796triColor_check-2.tif")
cell = io.imread(str(path_cell))
path = Path(f"images/raw/{FOLDER}/z-37_unmixed-red_EY2795WT-EY2796triColor_check-2.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes = 'c'
	image = nd2[0]
bkgd = neighbor_mean(image,cell)
clear = image - bkgd
clear[clear<0] = 0
gauss = filters.gaussian(clear,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/{FOLDER}/preprocessed_mitochondria_diploids3colors.tif",
	util.img_as_uint(gauss)
)
# LD
path_cell = Path(f"images/cell/{FOLDER}/camera-BF-after_EY2795triColor-EY2796WT_check-2.tif")
cell = io.imread(str(path_cell))
path = Path(f"images/raw/{FOLDER}/z-37_unmixed-red_EY2795triColor-EY2796WT_check-2.nd2")
with ND2Reader(str(path)) as nd2:
	nd2.bundle_axes = "zyx"
	nd2.iter_axes = 'c'
	image = nd2[1]
bkgd = neighbor_mean(image,cell)
clear = image - bkgd
clear[clear<0] = 0
gauss = filters.gaussian(clear,sigma=0.75,preserve_range=True).astype(int)
io.imsave(
	f"images/preprocessed/{FOLDER}/preprocessed_LD_diploids3colors.tif",
	util.img_as_uint(gauss)
)

# after Ilastik, Postprocess
# peroxisome, golgi, lipid droplet
for path in [
	Path(f"images/preprocessed/{FOLDER}/Probabilities_preprocessed_peroxisome_diploids3colors.tiff"),
	Path(f"images/preprocessed/{FOLDER}/Probabilities_preprocessed_golgi_diploids3colors.tiff"),
	Path(f"images/preprocessed/{FOLDER}/Probabilities_preprocessed_LD_diploids3colors.tiff")
]:
	img = io.imread(str(path))
	img = (img>0.5)
	path_ref = path.parent / f"{path.stem.partition('_')[2]}.tif"
	ref = io.imread(str(path_ref))
	output = segmentation.watershed(-ref,mask=img)
	io.imsave(
		f"images/labelled/{FOLDER}/{path_ref.name.partition('_')[2]}",
		util.img_as_uint(output)
	)
# mitochondria
path = Path(f"images/preprocessed/{FOLDER}/Probabilities_preprocessed_mitochondria_diploids3colors.tiff")
img = io.imread(str(path))
img = (img>0.5)
output = measure.label(img)
io.imsave(
	f"images/labelled/{FOLDER}/mitochondria_diploids3colors.tif",
	util.img_as_uint(output)
)
# ER
path = Path(f"images/preprocessed/{FOLDER}/Probabilities_preprocessed_ER_diploids3colors.tiff")
img = io.imread(str(path))
img = (img>0.5)
ske = skeletonize_zbyz(img)
io.imsave(
	f"images/labelled/{FOLDER}/ER_diploids3colors.tif",
	util.img_as_uint(ske)
)
# vacuole
path = Path(f"images/preprocessed/{FOLDER}/Probabilities_preprocessed_vacuole_diploids3colors.tiff")
img = io.imread(str(path))
img = (img>0.5)
path_cell = Path(f"images/cell/{FOLDER}/camera-BF-after_EY2795triColor-EY2796WT_check-2.tif")
cell = io.imread(path_cell)
ske = skeletonize_zbyz(img)
core = find_complete_rings(ske)
output = np.zeros_like(core,dtype=int)
for z in range(output.shape[0]):
	sample = core[z]
	candidates = np.unique(sample[cell>0])
	for color in candidates:
		if len(np.unique(cell[sample==color]))==1:
			output[z,sample==color] = color
io.imsave(
	f"images/labelled/{FOLDER}/vacuole_diploids3colors.tif",
	util.img_as_uint(output)
)

# Measure
organelles = [
	"peroxisome",
	"vacuole",
	"ER",
	"golgi",
	"mitochondria",
	"LD"
]
imgs_cell = {
	"peroxisome"  : "camera-BF-after_EY2795WT-EY2796triColor_check-2.tif",
	"vacuole"     : "camera-BF-after_EY2795triColor-EY2796WT_check-2.tif",
	"ER"          : "camera-BF-after_EY2795WT-EY2796triColor_check-2.tif",
	"golgi"       : "camera-BF-after_EY2795triColor-EY2796WT_check-2.tif",
	"mitochondria": "camera-BF-after_EY2795WT-EY2796triColor_check-2.tif",
	"LD"          : "camera-BF-after_EY2795triColor-EY2796WT_check-2.tif"
}
for organelle in organelles:
	path_cell      = Path(f"images/cell/{FOLDER}/{imgs_cell[organelle]}")
	path_organelle = Path(f"images/labelled/{FOLDER}/{organelle}_diploids3colors.tif")
	
	img_cell      = io.imread(str(path_cell))
	img_organelle = io.imread(str(path_organelle))

	results = []
	measured = {"organelle": organelle}
	for cell in measure.regionprops(img_cell):
		measured["cell-idx"] = cell.label
		measured["cell-area"] = cell.area

		min_row, min_col, max_row, max_col = cell.bbox
		img_cell_crop = cell.image
		img_orga_crop = img_organelle[:,min_row:max_row,min_col:max_col]
		for z in range(img_orga_crop.shape[0]):
			img_orga_crop[z] = img_orga_crop[z] * img_cell_crop
		if not organelle == "vacuole":
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
		measured = measured | measured_orga
		results.append(pd.DataFrame(measured))
	result = pd.concat(results,ignore_index=True)
	result.to_csv(f"data/results/{FOLDER}/{organelle}.csv",index=False)