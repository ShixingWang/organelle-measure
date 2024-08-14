# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from organelle_measure.data import read_results

# Idea: reample the volume fractions according to the fitted distribution from 
# MCMC simulation, and see how the centroids of conditions, and principal
# components will shift. This will strengthen our confidence of the error of 
# our conclusion.
