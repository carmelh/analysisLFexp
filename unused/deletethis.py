# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:42:37 2020

@author: chowe7
"""
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pyqtgraph as pg
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

def to_outline(roi):
    return np.logical_xor(ndimage.morphology.binary_dilation(roi),roi)


###############################################
################### INPUTS  ###################
###############################################MLA2_1x1_50ms_200pA_A_Stim_1
date = '190605'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra' + '\\' +date
currentFile = 'ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4' 
num_iterations=3
percentile = 90

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
################### decon ###################
###############################################


decStackA = gf.loadPickles(path,'\\deconvolvedStack_RL_infocus_{}'.format(num_iterations))
backgroundData_dec = gf.loadPickles(path,'\\deconvolvedBackgroundData_RL_infocus_{}'.format(num_iterations))    

# diff it folder    
decStackA = gf.loadPickles(path,'\\diff_it\\deconvolvedStack_RL_infocus_{}'.format(num_iterations))
backgroundData_dec=np.average(decStackA[:,10:30,10:30],axis=1)
backgroundData_dec=np.average(backgroundData_dec,axis=1)

# not renamed
decStackA = gf.loadPickles(path,'\\deconvolvedStack_infocus')
backgroundData_dec = gf.loadPickles(path,'\\deconvolvedBackgroundData_infocus')    

decStackA =[]
for ii in range(len(decon_mean_stack)):
    test = decon_mean_stack[ii]
    decStackA.append(test[round(len(decon_mean_stack[2])/2),:,:])
decStackA=np.array(decStackA)    
backgroundData_dec=np.average(decStackA[:,10:30,10:30],axis=1)
backgroundData_dec=np.average(backgroundData_dec,axis=1)

darkTrialData_dec=0
darkTrialAverage_dec =0




back_dec=np.mean(decStackA[:13,...],0)
df_dec=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)



keyword='deconvolved'
varImage_dec = np.var(df_dec,axis=-0)

plt.imshow(varImage_dec)

x1=30
x2=70
y1=35
y2=75
varImageROI_dec = varImage_dec[y1:y2,x1:x2]
plt.imshow(varImageROI_dec)

df_decROI = df_dec[:,y1:y2,x1:x2]

signalPixels_dec= np.array(np.where(varImageROI_dec > np.percentile(varImageROI_dec,percentile)))

trialData_dec = np.average(df_decROI[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)

binarized = 1.0 * (varImageROI_dec > np.percentile(varImageROI_dec,percentile))
outlineROI=to_outline(binarized)

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_dec,trialData_dec,darkTrialData_dec,13)
print(SNR)

# save to excel
fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,df.at[currentFile, 'LED power'],'',num_iterations,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
gf.appendCSV(cwd ,r'\stats_deconvolved_update2_{}'.format(date),fields)
    

plt.imshow(varImageROI_dec)
plt.imshow(outlineROI)



###############################################
################## refocused ##################
###############################################

reFstackA = gf.loadPickles(path,'\\refstack_infocus')

reFstackA =[]
for ii in range(len(refoc_mean_stack)):
    test = refoc_mean_stack[ii]
    reFstackA.append(test[round(len(refoc_mean_stack[2])/2),:,:])

reFstackA=np.array(reFstackA)    
darkTrialData_ref=90
darkTrialAverage_ref =90

backgroundData_ref=np.average(reFstackA[:,10:30,10:30],axis=1)
backgroundData_ref=np.average(backgroundData_ref,axis=1)


back_ref=np.mean(reFstackA[:13,...],0)
df_ref=100*(reFstackA-back_ref[None,...]) / (back_ref[None,...] - darkTrialAverage_ref)



keyword='refocussed'
varImage_ref = np.var(df_ref,axis=-0)

varImageROI_ref = varImage_ref[y1:y2,x1:x2]
df_refROI = df_ref[:,y1:y2,x1:x2]

trialData_ref = np.average(df_refROI[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)

outlineROI=to_outline(binarized)

# get stats
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_ref,trialData_ref,darkTrialData_ref,13)
print(SNR)

# save to excel
fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],stim,df.at[currentFile, 'LED power'],'',SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, fileNameDark, baselineDarkNoise]
gf.appendCSV(cwd ,r'\stats_refocused_update2_{}'.format(date),fields)
print('Saved Stats')










xS=7
xE=9
yS=0.06
yE=0.08

keyword='widefield_process'

time = np.arange(0, ts*len(processedTrace), ts)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(time,processedTrace,linewidth=4.0,color='k')
plt.plot([xS,xE],[yS,yS],linewidth=6.0,color='k')
plt.plot([xS,xS],[yS,yE],linewidth=6.0,color='k')
fig.set_size_inches(9,4)

#axis formatting
#ax.set_ylim((-0.003,0.08))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 

plt.tight_layout() 
plt.savefig(path + r'\\figures\\timeSeries_{}.png'.format(keyword), format='png', dpi=600)
plt.savefig(path + r'\\figures\\timeSeries_{}.eps'.format(keyword), format='eps', dpi=800)



  