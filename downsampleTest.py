# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:22:08 2021

@author: chowe7
"""

import numpy as np
import tifffile
import sys
import pandas as pd
import imagingAnalysis as ia
import idx_refocus as ref
import deconvolveLF as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import pyqtgraph as pg

stack= tifffile.imread(r'Y:\projects\thefarm2\live\Firefly\NIR-GECO_imaging\210325\slice1\area1\s1a1_WF_1P_1x1_200mA_100msExp_func_600frames_3\s1a1_WF_1P_1x1_200mA_100msExp_func_600frames_3_MMStack_Default.ome.tif')

from scipy import ndimage, misc
import matplotlib.pyplot as plt

result = ndimage.zoom(stack, 0.5)


start = perf_counter()
result2=np.zeros((600,512,512))
for time in range(len(stack)):
    result2[time,...]=ndimage.zoom(stack[time,...], 0.25)
end = perf_counter()
execution_time = (end - start)
print(execution_time)
#94.7595225999994

start = perf_counter() 
result=[]
for time in range(len(stack)):
    result.append(ndimage.zoom(stack[time,...], 0.25))

result=np.array(result)
end = perf_counter()
execution_time = (end - start)
print(execution_time)
#94.54463570000007

