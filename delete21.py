# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 09:20:38 2021

@author: chowe7
"""


import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import imagingAnalysis as ia
import idx_refocus as ref
import deconvolveLF as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf


cwd=r'Y:\projects\thefarm2\live\Firefly\NIR-GECO_imaging\210325\slice1\area1\WF_1P_1x1_200mA_100msExp_func_600frames_3'
filename=r'\WF_1P_1x1_200mA_100msExp_func_600frames_3_MMStack_Default.ome'

x,y= gf.importCSV(cwd,filename)

ts=100e-3
time = np.arange(0, ts*len(y), ts)

plt.plot(time,y)