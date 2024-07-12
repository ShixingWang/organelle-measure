# %%
import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util
from batch_apply import batch_apply

# %%
import javabridge
import bioformats
javabridge.start_vm(class_path=bioformats.JARS)

# %%
def resize(path_in,path_out):
	print(path_in.stem)
	image = np.empty((37,2044,2044))
	for z in range(37):
		plane = bioformats.load_image(str(path_in),z=z,rescale=False)
		image[z] = plane[:2044,:2044]
		print(f"    {z=},{plane.shape=}")
	io.imsave(
		str(path_out),
		util.img_as_float32(image)
	)	
	return None

# %%
list_in  = []
list_out = []
for path in Path("images/preprocessed/2024-06-25_2colorDiploidMeasure/NonSquarePre").glob("*.tif"):
	list_in.append(path)
	list_out.append(f"images/preprocessed/2024-06-25_2colorDiploidMeasure/{path.stem}.tif")
args = pd.DataFrame({
	"path_in" : list_in,
	"path_out": list_out
})
# %% 
batch_apply(resize,args)


# %%
def resize(path_in,path_out):
	print(path_in.stem)
	image = np.empty((2044,2044),dtype=int)
	plane = bioformats.load_image(str(path_in),z=0,rescale=False)
	image = plane[:2044,:2044]
	io.imsave(
		str(path_out),
		util.img_as_uint(image)
	)	
	return None

# %%
list_in  = []
list_out = []
for path in Path("images/cell/2024-06-25_2colorDiploidMeasure/NonSquare").glob("*.tif"):
	list_in.append(path)
	list_out.append(f"images/cell/2024-06-25_2colorDiploidMeasure/{path.stem}.tif")
args = pd.DataFrame({
	"path_in" : list_in,
	"path_out": list_out
})
# %% 
batch_apply(resize,args)


# %%
javabridge.kill_vm()
