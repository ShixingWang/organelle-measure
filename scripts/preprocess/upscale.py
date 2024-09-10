import numpy as np
import pandas as pd
from pathlib import Path
from skimage import io,util,transform,filters,segmentation,measure
from skimage.filters import rank
from organelle_measure.yeaz import yeaz_preprocesses,yeaz_label
from organelle_measure.tools import open_organelles,load_nd2_plane
from batch_apply import batch_apply

def upscaled_preprocess(path_in, path_out, organelle):
    img_raw   = open_organelles[organelle](str(path_in))
    img_up    = transform.rescale(img_raw,2,order=0,preserve_range=True)
    img_mean  = rank.mean(img_up,footprint=np.ones((3,3),dtype=int))
    img_gauss = filters.gaussian(img_mean,sigma=0.75,preserve_range=True)
    img_out = img_gauss.astype(int)
    io.imsave(
        str(path_out),
        util.img_as_uint(img_out)
    )
    return None

def upscaled_cell_segment(path_in,path_out):
    img_i = load_nd2_plane(str(path_in),frame='yx',axes='t',idx=0)
    for prep in yeaz_preprocesses:
        img_i = prep(img_i)
    img_b = yeaz_label(img_i,min_dist=5)
    img_b = segmentation.clear_border(img_b)
    properties = measure.regionprops(img_b)
    for prop in properties:
        if prop.area < 50: # hard coded threshold, bad
            img_b[img_b==prop.label] = 0
    img_b = measure.label(img_b)
    img_o = np.zeros((512,512),dtype=int) # hard coded size, bad
    shape0,shape1 = img_b.shape
    img_o[:shape0,:shape1] = img_b

    io.imsave(str(path_out),util.img_as_uint(img_o))
    print(f"...{path_out}")	
    return None

