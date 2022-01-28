# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 13:36:55 2020

@author: chowe7
"""

import numpy as np
import sys
import pandas as pd
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import pyqtgraph as pg
import matplotlib.pyplot as plt
import plotting_functions as pf

###############################################
################### INPUTS  ###################
###############################################
date = '190605'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra' + '\\' + date
currentFile = 'ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4'


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

x1=30
x2=70
y1=40
y2=80

percentile=98

###############################################
################## refocused ##################
###############################################

keyword='refocused'

reFstackA=gf.loadPickles(path,'\\refstack_infocus')

darkTrialData_ref=90
darkTrialAverage_ref =90

backgroundData_ref=np.average(reFstackA[:,10:30,10:30],axis=1)
backgroundData_ref=np.average(backgroundData_ref,axis=1)


back_ref=np.mean(reFstackA[:13,...],0)
df_ref=100*(reFstackA-back_ref[None,...]) / (back_ref[None,...] - darkTrialAverage_ref)


varImage_ref = np.var(df_ref,axis=-0)
plt.imshow(varImage_ref)

varImageROI_ref = varImage_ref[y1:y2,x1:x2]
df_refROI = df_ref[:,y1:y2,x1:x2]


signalPixels_ref= np.array(np.where(varImageROI_ref > np.percentile(varImageROI_ref,percentile)))

trialData_ref = np.average(df_refROI[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)

binarized = 1.0 * (varImageROI_ref > np.percentile(varImageROI_ref,percentile))
outlineROI=gf.to_outline(binarized)

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_ref,trialData_ref,darkTrialData_ref,13)
print(SNR)

# save to excel
fields=['Ref','-',SNR, peak_dF_F, df_noise, bleach, percentile, path]
gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_wProcess{}'.format(date),fields)
       

darkTrialData_dec=0
darkTrialAverage_dec =0
###############################################
################# deconvolved #################
###############################################
#num_iterations=21
num_iterations=[1,3,5,7,9,13,17,21]

for ii in range(len(num_iterations)):
    print(num_iterations[ii])
    decStackA=gf.loadPickles(path,'\\diff_it\\deconvolvedStack_RL_infocus_{}'.format(num_iterations[ii]))
      
    back_dec=np.mean(decStackA[:13,...],0)
    df_dec=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)
    
    varImage_dec = np.var(df_dec,axis=-0)
    varImageROI_dec = varImage_dec[y1:y2,x1:x2]
    df_decROI = df_dec[:,y1:y2,x1:x2]
        
    trialData_dec = np.average(df_decROI[:,signalPixels_ref[0],signalPixels_ref[1]], axis=1)
    
    backgroundData_dec=np.average(decStackA[:,10:30,10:30],axis=1)
    backgroundData_dec=np.average(backgroundData_dec,axis=1)
          
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_dec,trialData_dec,darkTrialData_dec,13)

    fields=['Dec',num_iterations[ii],SNR, peak_dF_F, df_noise, bleach, percentile, path]
    gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_wProcess{}'.format(date),fields)
        
#    ################### process ###################
#    trialDataDec_process = np.average(decStackA[:,51:57,43:48], axis=1)
#    trialDataDec_process = np.average(trialDataDec_process, axis=1)
#    processedTraceDec_process, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialDataDec_process, darkTrialData_dec, backgroundData_dec, stim)
#    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTraceDec_process,trialDataDec_process,darkTrialData_dec,baselineIdx)
##    print(SNR)
#    fields=['Dec',num_iterations[ii],SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, baselineDarkNoise]
#    gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_s1c2_process_{}'.format(date),fields)
#      
 
df_decT=np.zeros((42,200,101,101))    
for z in range(42):
    df_decT[z]=100*(decStackA[z]-back_dec[z]) / (back_dec[z] - darkTrialAverage_dec)



decon_RL_660=np.load(path + '\\diff_it\\decon_RL.npy')

num_iterations=[1,3,5,7,9,13,17,21]
darkTrialAverage_dec = 0

varImage_dec = np.var(df_dec,axis=-0)
signalPixels_dec = np.array(np.where(varImageROI_dec > np.percentile(varImageROI_dec,percentile)))

for ii in range(len(num_iterations)):
    print(num_iterations[ii])
    decStackA= decon_R_660L[ii,0:122,...]

    backgroundData_dec=np.average(decStackA[:,10:30,10:30],axis=1)
    backgroundData_dec=np.average(backgroundData_dec,axis=1)
    
    back_dec=np.mean(decStackA[:13,...],0)
    df_dec=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)
    
    varImage_dec = np.var(df_dec,axis=-0)

    varImageROI_dec = varImage_dec[y1:y2,x1:x2]
    df_decROI = df_dec[:,y1:y2,x1:x2]
        
    trialData_dec = np.average(df_decROI[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)
   
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_dec,trialData_dec,darkTrialData_dec,13)
    print(SNR)
    
    fields=['Dec',num_iterations[ii],SNR, peak_dF_F, df_noise, bleach, percentile,path]
    gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_wProcess{}'.format(date),fields)
        
    ################### process ###################
    trialDataDec_process = np.average(decStackA[:,51:57,43:48], axis=1)
    trialDataDec_process = np.average(trialDataDec_process, axis=1)
    processedTraceDec_process, diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialDataDec_process, darkTrialData_dec, backgroundData_dec, stim)
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTraceDec_process,trialDataDec_process,darkTrialData_dec,baselineIdx)
#    print(SNR)
    fields=['Dec',num_iterations[ii],SNR, peak_dF_F, df_noise, bleach]
    gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_s1c2_process_{}'.format(date),fields)
      
    


x1=30-3
x2=70+3
y1=40-3
y2=80+3
varImageROI_dec = varImage_dec[y1:y2,x1:x2]
df_decROI = df_dec[:,y1:y2,x1:x2]
df_refROI = df_ref[:,y1:y2,x1:x2]

binarized = 1.0 * (varImageROI_dec > np.percentile(varImageROI_dec,percentile))
outlineROI=gf.to_outline(binarized)

lengthSB = 25
pixelSize = 5


my_cmap = cm.hsv
my_cmap.set_under('k', alpha=0)
fig, ax = plt.subplots()
ax.imshow(df_refROI[17,...], cmap=cm.viridis)

ax.imshow(outlineROI*1, cmap=cm.hsv, interpolation='none',clim=[0.9, 1])
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
forceAspect(ax,aspect=1)
plt.tight_layout()  
pf.saveFigure(fig,analysisFolder,'ref_wSB-25um')


plt.imshow(df_refROI[17,...])
plt.imshow(test)
outlineROI


analysisFolder = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4\figures'



minVR=np.min(df_ref[:,40,33:103,25:95])
maxVR= np.max(df_ref[:,40,33:103,25:95])

minVD=np.min(df_dec[:,40,33:103,25:95])
maxVD= np.max(df_dec[:,40,33:103,25:95])
for tp in range(200):
    print(tp)
    number_str = str(tp)
    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.imshow(df_ref[tp,40,33:103,25:95],cmap='gray',vmin = minVR, vmax =maxVR)
    plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

    #axis formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    forceAspect(ax,aspect=1)
    plt.tight_layout()  
    plt.savefig(analysisFolder + r'\\timeSeries\\refoc\\tp-grey-{}.png'.format(number_str.zfill(3)), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)   
    
    
    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.imshow(df_dec[tp,40,33:103,25:95],cmap='gray',vmin = minVD, vmax =maxVD)
    plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

    #axis formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    forceAspect(ax,aspect=1)
    plt.tight_layout()  
    plt.savefig(analysisFolder + r'\\timeSeries\\decon\\tp-grey-{}.png'.format(number_str.zfill(3)), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)  
    
    
    
xS = 57
y = 62
    
fig = pf.plotImage(np.var(df_ref[:,40,33:103,25:95],axis=0))
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
pf.saveFigure(fig,analysisFolder,'ref_wSB-25um')
