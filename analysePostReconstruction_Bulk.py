# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 13:45:44 2022

@author: chowe7
"""

%reset -f


import numpy as np
import sys
import pandas as pd
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import pyqtgraph as pg
import matplotlib.pyplot as plt
import plotting_functions as pf
import scipy.ndimage as ndimage

def to_outline(roi):
    return np.logical_xor(ndimage.morphology.binary_dilation(roi),roi)


###############################################
################### INPUTS  ###################
###############################################
date = '191105'
#cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk' + '\\' + date
#currentFile = 'LF_1x1_50mA_50ms_A-Stim_3'
path= r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk\191105\slice2\cell1\LF_1x1_50ms-50pA__1'

###############################################
###################   REF    #################
###############################################

refoc_mean_stack=np.load(path + '\\stack_refoc\\refocused' + '\\' + 'refoc_mean_stack.npy')
#reFstackA=refoc_mean_stack[:,round(len(refoc_mean_stack[1])/2),...]
darkTrialAverage_ref =90
inFocus_ref=round(len(refoc_mean_stack[1])/2)

back_ref=np.mean(refoc_mean_stack[:13,...],0)
df_ref=100*(refoc_mean_stack-back_ref[None,...]) / (back_ref[None,...] - darkTrialAverage_ref)

varImage_ref = np.var(df_ref,axis=-0)

#BULK
x1=10
x2=90
y1=10
y2=90
y1=10
y2=90

varImageROI_ref = varImage_ref[:,y1:y2,x1:x2]
pg.image(varImageROI_ref)

df_refROI = df_ref[:,:,y1:y2,x1:x2]



###############################################
#################### decon# ###################
###############################################

decon_RL_660=np.load(path + '\\stack_refoc\\deconvolved' + '\\' + '{}_decon_RL_PSF-660.npy'.format(currentFile))

num_iterations=[1,3,5,7,9,13,17,21]
darkTrialAverage_dec = 0


#all depths
df_dec=np.zeros((len(decon_RL_660[1]),len(decon_RL_660[1][1]),101,101))
for depth in range(len(decon_RL_660[1])):
    decStackA=decon_RL_660[1,depth,...]
    back_dec=np.mean(decStackA[:13,...],0)
    df_dec[depth]=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)
varImage_dec = np.var(df_dec,axis=1)              
      
pg.image(varImage_dec[:,y1:y2,x1:x2])

pg.image(df_dec[40,:,y1:y2,x1:x2])

plt.imshow(varImageROI_ref[inFocus_ref])

# BULK     
df_IF_ref=df_refROI[:,inFocus_ref,...]

trialData_ref = np.average(df_IF_ref[:,36:38,61:63], axis=1)
trialData_ref = np.average(trialData_ref, axis=1) 
trialData_ref=trialData_ref+0.5
%varexp --plot trialData_ref

peakIdx =44
baselineIdx=30

baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(peakIdx,trialData_ref,trialData_ref,darkTrialAverage_ref,baselineIdx)
print(SNR,peak_dF_F, df_noise)

fields=['Ref','-','cell5',SNR, peak_dF_F, df_noise, bleach,path,peakIdx,baselineIdx]
gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk\{}'.format(date),r'\stats_660nmPSF_SPmanual'.format(date),fields)


# Dec
varImageROI_dec = varImage_dec[:,y1:y2,x1:x2]
df_decROI = df_dec[:,:,y1:y2,x1:x2]
df_IF_dec=df_decROI[39,:,...]
trialData_dec = np.average(df_IF_dec[:,36:38,61:63], axis=1)
trialData_dec = np.average(trialData_dec, axis=1) 
trialData_dec=trialData_dec+0.5
%varexp --plot trialData_dec

baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(peakIdx,trialData_dec,trialData_dec,darkTrialAverage_dec,baselineIdx)
print(SNR,peak_dF_F, df_noise)

fields=['Dec','3','cell5',SNR, peak_dF_F, df_noise, bleach,path,peakIdx,baselineIdx]
gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Bulk\{}'.format(date),r'\stats_660nmPSF_SPmanual'.format(date),fields)







ts=1/20

# FIGURE
time = np.arange(0, ts*len(trialData_ref), ts)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(time[0:175],trialData_ref[0:175]+10,linewidth=3.0,color='k')
plt.plot(time[0:175],trialData_dec[0:175],linewidth=3.0,color='k')
plt.plot([5.8,5.8],[5,8],linewidth=4.0,color='k')
plt.plot([5.8,7.8],[5,5],linewidth=4.0,color='k')
fig.set_size_inches(6,6)

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.savefig(figureFolder + r'\\timeSeries_cell5_RefDec_3per_2sec.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\timeSeries_cell5_RefDec_3per_2sec.eps', format='eps', dpi=1900, bbox_inches='tight')

