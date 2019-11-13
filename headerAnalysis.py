# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:46:28 2019

@author: chowe7
"""
# to do:LED power, skip dark image sometimes, timer on 

import numpy as np
import tifffile
import sys
import pandas as pd
#sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia
#sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import idx_refocus as ref
import deconvolveLF as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf


###############################################
################### INPUTS  ###################
###############################################
date = '191105'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk\191105'
currentFile = 'LF_1x1_50ms-50pA__1' 
num_iterations=3
#fileNameDark = r'\MLA2_1x1_50ms_150pA_A-STIM_DARK_1\MLA2_1x1_50ms_150pA_A-STIM_DARK_1_MMStack_Pos0.ome.tif'


###############################################
################### setup ###################
###############################################

data_summary = pd.ExcelFile(cwd+r'\result_summary_{}.xlsx'.format(date))
df = data_summary.parse('Sheet1')    

sliceNo = r'\slice{}'.format(df.at[currentFile, 'slice'])
cellNo = r'\cell{}'.format(df.at[currentFile, 'cell'])

stim = df.at[currentFile, 'Stim Prot']
ts=df.at[currentFile, 'Exp Time']
fs=1/ts 

fileName = r'\{}_MMStack_Pos0.ome.tif'.format(currentFile)
trialFolder = r'\{}'.format(currentFile)
path = cwd + sliceNo + cellNo + trialFolder
pathDarkTrial = cwd + sliceNo + cellNo + r'\{}'.format(df.at[currentFile, 'Dark folder'])
fileNameDark = r'\{}'.format(df.at[currentFile, 'Dark file'])

###############################################
################### refocus ###################
###############################################
if df.at[currentFile, 'Imaging'] == 'MLA':
    r,center = (np.array([df.at[currentFile, 'Rdy'],df.at[currentFile, 'Rdx']]),np.array([df.at[currentFile, 'y'],df.at[currentFile, 'x']])) 

    stack = tifffile.imread(path + fileName)
    #stackDark = tifffile.imread(cwd + sliceNo + cellNo + fileNameDark)
    print('stacks loaded')
    
    keyword='refocussed'
    
    trialData, varImage, backgroundData, darkTrialData, signalPixels = ref.main(stack,r,center,path,pathDarkTrial,fileNameDark)
    
    # process trace
    processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
    print('Finished Processing')
    
    # get stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)
    
    # save to excel
    fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,df.at[currentFile, 'LED power'],'',SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
    gf.appendCSV(cwd ,r'\stats_refocussed_{}'.format(date),fields)
    print('Saved Stats')
    
    
    df.at[currentFile, 'Refoc'] = 1
    print('Finished Refocussing')
    
    
    ##################################################
    ################### Deconvolve ###################
    ##################################################
    print('Starting Deconvolution')
    keyword = 'deconvolved'
    
    trialData, varImage, backgroundData, darkTrialData = dlf.getDeconvolution(stack,r,center,num_iterations,path,pathDarkTrial,fileNameDark,signalPixels)
    
    # process trace
    processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
    print('Finished Processing')
    
    # get stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)

    # save to excel
    fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,df.at[currentFile, 'LED power'],'',num_iterations,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
    gf.appendCSV(cwd ,r'\stats_deconvolved_{}'.format(date),fields)
    
    print('Saved Stats')
    
    df.at[currentFile, 'Decon'] = 1

elif df.at[currentFile, 'Imaging'] == 'WF':
    x,trialData = gf.importCSV(path,'\{}_MMStack_Pos0.ome'.format(currentFile))
    x,backgroundData = gf.importCSV(path,'\{}_MMStack_Pos0.ome_BG'.format(currentFile))
    x,darkTrialData = gf.importCSV(pathDarkTrial,fileNameDark)
    
    # process trace
    processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
    print('Finished Processing')
    
    # get stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)
    
    # save to excel
    fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,df.at[currentFile, 'LED power'],'',SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
    gf.appendCSV(cwd ,r'\stats_WF_{}'.format(date),fields)
    print('Saved Stats')
    
    df.at[currentFile, 'WF'] = 1