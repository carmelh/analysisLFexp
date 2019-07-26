# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:46:28 2019

@author: chowe7
"""
# to do:
#   get MLA locations from excel
#   get filenames from excel

import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
import tifffile
import time
import csv
import sys
sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia

font = {'family': 'sans',
        'weight': 'normal',
        'size': 16,
        }

plt.rc('font',**font)

r,center = (np.array([0.87,19.505]),np.array([1024.3,1015.4])) 

cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724'
sliceNo = r'\slice1'
cellNo = r'\cell1'
fileName = r'\MLA3_1x1_50ms_150pA_A-STIM_1\MLA3_1x1_50ms_150pA_A-STIM_1_MMStack_Pos0.ome.tif'
fileNameDark = r'\MLA2_1x1_50ms_150pA_A-STIM_DARK_1\MLA2_1x1_50ms_150pA_A-STIM_DARK_1_MMStack_Pos0.ome.tif'

Stim = 'A Stim'

#    cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724\slice1\cell1'
 #   stack = tifffile.imread(cwd + r'\MLA3_1x1_50ms_150pA_A-STIM_1\MLA3_1x1_50ms_150pA_A-STIM_1_MMStack_Pos0.ome.tif')

reFstack = []
for ii in range(len(stack)):
    im = stack[ii,...] 
    print('Stack loaded...',ii)    
    result = get_refocussed(im,r,center,np.arange(-10,10,1),n_views = 19)
    print('Refocussed.')
    reFstack.append(result[10,:,:])
    #pg.image(result)
  
# auto find pixels containing signal    
reFstackA=np.array(reFstack)
varImage = np.var(reFstackA,axis=-0)
plt.imshow(varImage)
signalPixels = np.array(np.where(varImage > np.percentile(varImage,99.92)))
trialData = np.average(reFstackA[:,signalPixels[0],signalPixels[1]], axis=1)    
   
backgroundData=np.average(reFstackA[:,10:30,10:30],axis=1)
backgroundData=np.average(backgroundData,axis=1)

    
#darkstack        
reStackDark = []
for ii in range(len(stackDark)):
    im = stackDark[ii,...] 
    print('Stack loaded...',ii)    
    result = get_refocussed(im,r,center,np.arange(-10,10,1),n_views = 19)
    print('Refocussed.')
    reStackDark.append(result[10,:,:])
    
darkTrialData=[]
for jj in range(len(reStackDark)):
    x=reStackDark[jj]
    d=np.average(x[60:63,42:44])
    darkTrialData.append(d)
    
baselineIdx = 11 # for A Stim

# process trace
processedTrace, diffROI,processedBackgroundTrace = ia.processRawTrace(trialData, darkTrialData, backgroundData, baselineIdx)

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach= ia.getStatistics(processedTrace, trialData, darkTrialData, baselineIdx)

# save to excel
#fields=['slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
fields=[sliceNo, cellNo, fileName,Stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
with open(cwd + r'\stats.csv', 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\r')
    writer.writerow(fields)