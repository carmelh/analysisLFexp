# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:46:28 2019

@author: chowe7
"""
# to do:
#   get MLA locations from excel
#   get filenames from excel
#   get stim and freq from excel

import numpy as np
import matplotlib.pyplot as plt
#import pyqtgraph as pg
import tifffile
import time
import csv
import sys
#sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia
#sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import idx_refocus as ref
import deconvolveLF as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import plotting_functions as pf

r,center = (np.array([0.87,19.505]),np.array([1024.3,1015.4])) 

cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724'
sliceNo = r'\slice1'
cellNo = r'\cell1'
fileName = r'\MLA3_1x1_50ms_150pA_A-STIM_1_MMStack_Pos0.ome.tif'
trialFolder = r'\MLA3_1x1_50ms_150pA_A-STIM_1'
path = cwd + sliceNo + cellNo + trialFolder

fileNameDark = r'\MLA2_1x1_50ms_150pA_A-STIM_DARK_1\MLA2_1x1_50ms_150pA_A-STIM_DARK_1_MMStack_Pos0.ome.tif'

stack = tifffile.imread(path + fileName)
stackDark = tifffile.imread(cwd + sliceNo + cellNo + fileNameDark)

Stim = 'A Stim'
fs=20 
ts=1/fs

#    cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724\slice1\cell1'
 #   stack = tifffile.imread(cwd + r'\MLA3_1x1_50ms_150pA_A-STIM_1\MLA3_1x1_50ms_150pA_A-STIM_1_MMStack_Pos0.ome.tif')
 
###############################################
################### refocus ###################
###############################################
keyword='refocussed'

trialData, varImage, backgroundData, darkTrialData = ref.main(stack,stackDark,r,center,path)
plt.imshow(varImage)

baselineIdx = 11 # for A Stim

# process trace
processedTrace, diffROI,processedBackgroundTrace = ia.processRawTrace(trialData, darkTrialData, backgroundData, baselineIdx)
print('Finished Processing')

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)

# save to excel
#fields=['slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
fields=[sliceNo, cellNo, fileName,Stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
with open(cwd + sliceNo + r'\stats_refocussed.csv', 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\r')
    writer.writerow(fields)
print('Saved Stats')


# first plot to figure out length and location of scale bars
plt.plot()

xS=7
xE=xS+2
yS=0.005
yE=yS+0.005
pf.plotTimeData(ts,processedTrace,xS,xE,yS,yE,path,keyword)



##################################################
################### Deconvolve ###################
##################################################
keyword = 'deconvolved'

trialData, varImage, backgroundData, darkTrialData = dlf.getDeconvolution(stack,stackDark,path)
plt.imshow(varImage)

# process trace
processedTrace, diffROI,processedBackgroundTrace = ia.processRawTrace(trialData, darkTrialData, backgroundData, baselineIdx)
print('Finished Processing')

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)

# save to excel
#fields=['slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
fields=[sliceNo, cellNo, fileName,Stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
with open(cwd + sliceNo + r'\stats_deconvolved.csv', 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\r')
    writer.writerow(fields)
print('Saved Stats')