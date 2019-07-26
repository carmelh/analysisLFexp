# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:46:28 2019

@author: chowe7
"""
# to do:LED power, skip dark image sometimes, timer on 

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
import pandas as pd



###############################################
################### INPUTS  ###################
###############################################

cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724'
currentFile = 'MLA2_1x1_50ms_150pA_A-STIM_4' 

fileNameDark = r'\MLA2_1x1_50ms_150pA_A-STIM_DARK_1\MLA2_1x1_50ms_150pA_A-STIM_DARK_1_MMStack_Pos0.ome.tif'

#fileName = r'\MLA3_1x1_50ms_150pA_A-STIM_1_MMStack_Pos0.ome.tif'

###############################################
################### setup ###################
###############################################

data_summary = pd.ExcelFile(cwd+r'\result_summary.xlsx')
df = data_summary.parse('Sheet1')    

sliceNo = r'\slice{}'.format(df.at[currentFile, 'slice'])
cellNo = r'\cell{}'.format(df.at[currentFile, 'cell'])

stim = df.at[currentFile, 'Stim Prot']
ts=df.at[currentFile, 'Exp Time']
fs=1/ts 

r,center = (np.array([df.at[currentFile, 'Rdy'],df.at[currentFile, 'Rdx']]),np.array([df.at[currentFile, 'y'],df.at[currentFile, 'x']])) 

fileName = r'\{}_MMStack_Pos0.ome.tif'.format(currentFile)
trialFolder = r'\{}'.format(currentFile)
path = cwd + sliceNo + cellNo + trialFolder

stack = tifffile.imread(path + fileName)
stackDark = tifffile.imread(cwd + sliceNo + cellNo + fileNameDark)


###############################################
################### refocus ###################
###############################################
keyword='refocussed'

trialData, varImage, backgroundData, darkTrialData, signalPixels = ref.main(stack,stackDark,r,center,path)
plt.imshow(varImage)

#baselineIdx = 11 # for A Stim

# process trace
processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
print('Finished Processing')

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)

# save to excel
#fields=['slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
with open(cwd + sliceNo + r'\stats_refocussed.csv', 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\r')
    writer.writerow(fields)
print('Saved Stats')


# first plot to figure out length and location of scale bars
#plt.plot()
#
#xS=7
#xE=xS+2
#yS=0.005
#yE=yS+0.005
#pf.plotTimeData(ts,processedTrace,xS,xE,yS,yE,path,keyword)

df.at[currentFile, 'Refoc'] = 1

##################################################
################### Deconvolve ###################
##################################################
keyword = 'deconvolved'

trialData, varImage, backgroundData, darkTrialData = dlf.getDeconvolution(stack,stackDark,path,signalPixels)
plt.imshow(varImage)

# process trace
processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
print('Finished Processing')

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)

# save to excel
#fields=['slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
with open(cwd + sliceNo + r'\stats_deconvolved.csv', 'a', newline='') as f:
    writer = csv.writer(f, lineterminator='\r')
    writer.writerow(fields)
print('Saved Stats')


# first plot to figure out length and location of scale bars
#plt.plot()
#
#xS=7
#xE=xS+2
#yS=0.005
#yE=yS+0.005
#pf.plotTimeData(ts,processedTrace,xS,xE,yS,yE,path,keyword)

df.at[currentFile, 'Decon'] = 1