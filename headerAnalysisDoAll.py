# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:46:28 2019

@author: chowe7
"""
# to do:LED power, skip dark image sometimes

import numpy as np
import tifffile
import sys
import pandas as pd
import imagingAnalysis as ia
import idx_refocus as ref
import deconvolveLF as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf


###############################################
################### INPUTS  ###################
###############################################

cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724'
fileNameDark = r'\MLA2_1x1_50ms_150pA_A-STIM_DARK_1\MLA2_1x1_50ms_150pA_A-STIM_DARK_1_MMStack_Pos0.ome.tif'

data_summary = pd.ExcelFile(cwd+r'\result_summary.xlsx')
df = data_summary.parse('Sheet1')    

statsExcelRe = 0
statsExcelDe = 0
statsExcelWF = 0


for trial in range(len(df)):
    currentFile = df.index.values[trial]
    print('Starting file {}'.format(currentFile))
    
    ###############################################
    #################### setup ####################
    ###############################################
    
    sliceNo = r'\slice{}'.format(df.at[currentFile, 'slice'])
    cellNo = r'\cell{}'.format(df.at[currentFile, 'cell'])
    
    stim = df.at[currentFile, 'Stim Prot']
    ts=df.at[currentFile, 'Exp Time']
    fs=1/ts 
    
    fileName = r'\{}_MMStack_Pos0.ome.tif'.format(currentFile)
    trialFolder = r'\{}'.format(currentFile)
    path = cwd + sliceNo + cellNo + trialFolder
    
    ###############################################
    #################### USABLE ###################
    ################## WF or LF? ##################
    ###############################################
    if df.at[currentFile, 'Usable'] == 'Y':
        if df.at[currentFile, 'Imaging'] == 'MLA':
            ###############################################
            ################### refocus ###################
            ###############################################
            r,center = (np.array([df.at[currentFile, 'Rdy'],df.at[currentFile, 'Rdx']]),np.array([df.at[currentFile, 'y'],df.at[currentFile, 'x']])) 
        
            stack = tifffile.imread(path + fileName)
            stackDark = tifffile.imread(cwd + sliceNo + cellNo + fileNameDark)
            print('stacks loaded')
            
            keyword='refocussed'
            
            trialData, varImage, backgroundData, darkTrialData, signalPixels = ref.main(stack,stackDark,r,center,path)
            #plt.imshow(varImage)
            
            # process trace
            processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
            print('Finished Processing')
            
            # get stats
            baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)
            
            # save to excel
            if statsExcelRe == 0:
                fields=['','slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
                gf.appendCSV(cwd + sliceNo,r'\stats_refocussed.csv',fields)  
                statsExcelRe = 1
                
            fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
            gf.appendCSV(cwd + sliceNo,r'\stats_refocussed.csv',fields)
            print('Saved Stats')
            
            df.at[currentFile, 'Refoc'] = 1
            print('Finished Refocussing')
            
            
            ##################################################
            ################### Deconvolve ###################
            ##################################################
            print('Starting Deconvolution')
            keyword = 'deconvolved'
            
            trialData, varImage, backgroundData, darkTrialData = dlf.getDeconvolution(stack,stackDark,r,center,path,signalPixels)
            
            # process trace
            processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
            print('Finished Processing')
            
            # get stats
            baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)
            
            # save to excel
            if statsExcelDe == 0:
                fields=['','slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
                gf.appendCSV(cwd + sliceNo,r'\stats_deconvolved.csv',fields)
                statsExcelDe = 1
                
            fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
            gf.appendCSV(cwd + sliceNo,r'\stats_deconvolved.csv',fields)
            print('Saved Stats')
            
            df.at[currentFile, 'Decon'] = 1
        
        elif df.at[currentFile, 'Imaging'] == 'WF':
            x,trialData = gf.importCSV(path,'\{}_MMStack_Pos0.ome'.format(currentFile))
            x,backgroundData = gf.importCSV(path,'\{}_MMStack_Pos0.ome_BG'.format(currentFile))
            x,darkTrialData = gf.importCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724\slice1\cell1\MLA2_1x1_50ms_150pA_A-STIM_DARK_1','\MLA2_1x1_50ms_150pA_A-STIM_DARK_1_MMStack_Pos0.ome')
            
            # process trace
            processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
            print('Finished Processing')
            
            # get stats
            baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTrace,trialData,darkTrialData,baselineIdx)
            
            # save to excel
            if statsExcelWF == 0: 
                fields=['','slice num', 'cell num', 'fileName','Stim','SNR','baseline', 'baseline photons', 'baseline noise', 'peak signal', 'pk photons', 'peak dF/F', 'df noise', 'bleach', 'Dark filename', 'baseline dark noise']
                gf.appendCSV(cwd + sliceNo,r'\stats_WF',fields)
                statsExcelWF = 1
                
            fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
            gf.appendCSV(cwd + sliceNo,r'\stats_WF',fields)
            print('Saved Stats')
            
            df.at[currentFile, 'WF'] = 1
