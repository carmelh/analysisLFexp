# -*- coding: utf-8 -*-
"""
Created on Wed May 20 09:12:06 2020

@author: chowe7


averaging trials
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf


###############################################
################### INPUTS  ###################
###############################################
date = '190724'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724'
num_iterations=3

analysisPath = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\TemporalFWHM'
fileNameDark = r'\MLA2_1x1_50ms_150pA_A-STIM_DARK_1'
stim = 'A Stim'


###############################################
################### Ref ###################
###############################################

df = pd.read_csv(cwd+r'\stats_refocussed_{}.csv'.format(date))
df = df.set_index('file name')

trials_ref=[]
for trial in range(len(df)):
    
    currentFile = df.index.values[trial]
    sliceNo = r'\slice{}'.format(df.at[currentFile, 'slice'])
    cellNo = r'\cell{}'.format(df.at[currentFile, 'cell'])   
    
    trialFolder = r'\{}'.format(currentFile)
    path = cwd + sliceNo + cellNo + trialFolder
    
    pathDarkTrial = cwd + sliceNo + cellNo + fileNameDark
    print('Starting file {}'.format(currentFile))


    trialData = gf.loadPickles(path,'\\refocussedTrialData_infocus')
    backgroundData = gf.loadPickles(path,'\\refocussedBackgroundData_infocus')

    if trial == 0:
        darkTrialData = gf.loadPickles(pathDarkTrial,'\\refDarkTrialData_infocus')

    processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
    plt.plot(processedTrace)
    trials_ref.append(processedTrace)
    try:
        darkTrialData = gf.loadPickles(pathDarkTrial,'\\refDarkTrialData_infocus')

        processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
        plt.plot(processedTrace)
        trials_ref.append(processedTrace)
    
    except:
        stim = 'A Stim'        
        
        
average_ref= np.average(trials_ref,axis=0)

ts=1/20
time = np.arange(0, ts*len(processedTrace), ts)

fig = plt.gcf()   
ax = plt.subplot(111)   
for ii in range(len(trials_ref)):
    plt.plot(time,gf.norm(trials_ref[ii]),linewidth=3.0,color='darkgray')
plt.plot(time,gf.norm(average_ref),linewidth=3.0,color='#049BEC')
plt.plot([2.2,2.7],[0.9,0.9],linewidth=4.0,color='k')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()
fig.set_size_inches(10,4)
plt.xlim([0.25,2.75])
plt.savefig(analysisPath + r'\\averagedTimeSeriesRef_{}.png'.format(date), format='png', dpi=600)
plt.savefig(analysisPath + r'\\averagedTimeSeriesRef_{}.eps'.format(date), format='eps', dpi=800)
plt.close(fig)  

    

###############################################
################### Decon ###################
###############################################

df = pd.read_csv(cwd+r'\stats_deconvolved_{}.csv'.format(date))
df = df.set_index('file name')

trials=[]
for trial in range(len(df)):
    
    trial=3
    currentFile = df.index.values[trial]
    sliceNo = r'\slice{}'.format(df.at[currentFile, 'slice num'])
    cellNo = r'\cell{}'.format(df.at[currentFile, 'cell num'])   
    
    fileName = r'\{}_MMStack_Pos0.ome.tif'.format(currentFile)
    trialFolder = r'\{}'.format(currentFile)
    path = cwd + sliceNo + cellNo + trialFolder
    
    pathDarkTrial = cwd + sliceNo + cellNo + fileNameDark
    print('Starting file {}'.format(currentFile))



    decStackA = gf.loadPickles(path,'\\deconvolvedStack_infocus'.format(num_iterations))
    trialData = gf.loadPickles(path,'\\deconvolvedTrialData_infocus')
    backgroundData = gf.loadPickles(path,'\\deconvolvedBackgroundData_infocus')   



    decStackA = gf.loadPickles(path,'\\deconvolvedStack_infocus'.format(num_iterations))
    backgroundData = gf.loadPickles(path,'\\deconvolvedBackgroundData_RL_infocus_{}'.format(num_iterations))   
    darkTrialData = gf.loadPickles(pathDarkTrial,'\\deconvolvedDarkTrialData_RL_infocus_{}'.format(num_iterations))
    trialData = np.average(decStackA[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)  

    stim = 'A Stim'
    processedTrace, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
    plt.plot(processedTrace)
    trials.append(processedTrace)


    
    try:
        decStackA = gf.loadPickles(path,'\\deconvolvedStack_RL_infocus_{}'.format(num_iterations))
        trialData = gf.loadPickles(path,'\\deconvolvedTrialData_infocus')
        backgroundData = gf.loadPickles(path,'\\deconvolvedBackgroundData_RL_infocus_{}'.format(num_iterations))   
    

    except:
        decStackA = gf.loadPickles(path,'\\diff_it\\deconvolvedStack_RL_infocus_{}'.format(num_iterations))
        trialData = gf.loadPickles(path,'\\diff_it\\deconvolvedTrialData_RL_infocus_{}'.format(num_iterations))
        backgroundData = gf.loadPickles(path,'\\diff_it\\deconvolvedBackgroundData_RL_infocus_{}'.format(num_iterations)) 
        
        


average_dec= np.average(trials,axis=0)


fig = plt.gcf()   
ax = plt.subplot(111)   
for ii in range(len(trials)):
    plt.plot(time,gf.norm(trials[ii]),linewidth=3.0,color='darkgray')
plt.plot(time,gf.norm(average_dec),linewidth=3.0,color='r')
plt.plot([2.2,2.7],[0.9,0.9],linewidth=4.0,color='k')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()
fig.set_size_inches(10,4)
plt.xlim([0.25,2.75])
plt.savefig(analysisPath + r'\\averagedTimeSeriesDec_{}.png'.format(date), format='png', dpi=600)
plt.savefig(analysisPath + r'\\averagedTimeSeriesDec_{}.eps'.format(date), format='eps', dpi=800)
plt.close(fig)  

