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
sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia
import idx_refocus_new as ref
import deconvolveLF_new as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf

###############################################
################### INPUTS  ###################
###############################################
date = '210325'
cwd = r'Y:\projects\thefarm2\live\Firefly\NIR-GECO_imaging' + '\\' + date
currentFile = 's1a1_LF_1P_1x1_50mA_100msExp_stack_5um_1'
num_iterations=3

###############################################
################### setup ###################
###############################################

df = pd.read_csv(cwd+r'\results_summary_{}.csv'.format(date),index_col=1)

ts=df.at[currentFile, 'timePeriod (s)']
fs=1/ts 

fileName = r'\{}_MMStack_Pos0.ome.tif'.format(currentFile)
trialFolder = r'\{}'.format(currentFile)
path = df.at[currentFile, 'main folder'] + trialFolder

depths = np.arange(-40,40,1)

baselineIdx=13

###############################################
################### refocus ###################
###############################################
if df.at[currentFile, 'WF/LF'] == 'LF':
    r,center = (np.array([df.at[currentFile, 'Rdy'],df.at[currentFile, 'Rdx']]),np.array([df.at[currentFile, 'y'],df.at[currentFile, 'x']])) 

    stack = tifffile.imread(path + fileName)
    #stackDark = tifffile.imread(cwd + sliceNo + cellNo + fileNameDark)
    print('stacks loaded')
    
    keyword='refocused'
    refoc_mean_stack=ref.main(stack,r,center,depths,path)
    darkTrial=90
    
    delta_f = ia.to_df(refoc_mean_stack,darkTrial,baselineIdx)
    
    # process trace
    processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
    print('Finished Processing')
    
    # get stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)
    
    # save to excel
    fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,df.at[currentFile, 'LED power'],'',SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
    gf.appendCSV(cwd ,r'\stats_refocused_{}'.format(date),fields)
    print('Saved Stats')
    
    
    df.at[currentFile, 'Refoc'] = 1
    print('Finished Refocussing')
    
    
    ##################################################
    ################### Deconvolve ###################
    ##################################################
    print('Starting Deconvolution')
    keyword = 'deconvolved'
    darkTrial=0
    decon_mean_stack = dlf.getDeconvolution(stack,r,center,num_iterations,depths,path)
    
    
    
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

elif df.at[currentFile, 'WF/LF'] == 'WF':
    stack = tifffile.imread(path + fileName)
    print('stacks loaded')
    
    keyword='widefield'
    
#    x,trialData = gf.importCSV(path,'\{}_MMStack_Pos0.ome'.format(currentFile))
#    x,backgroundData = gf.importCSV(path,'\{}_MMStack_Pos0.ome_BG'.format(currentFile))
#    x,darkTrialData = gf.importCSV(pathDarkTrial,fileNameDark)
    
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