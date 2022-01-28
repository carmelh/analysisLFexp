# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:46:29 2021

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
date = '190730'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra' + '\\' + date
currentFile = 'MLA2_1x1_50ms_200pA_A_Stim_1'



###############################################
################### setup ###################
###############################################

data_summary = pd.ExcelFile(cwd+r'\result_summary_{}.xlsx'.format(date))
df = data_summary.parse('Sheet1')    
path = cwd + r'\slice{}'.format(df.at[currentFile, 'slice']) + r'\cell{}'.format(df.at[currentFile, 'cell']) + r'\{}'.format(currentFile)
fileName = r'\{}_MMStack_Pos0.ome.tif'.format(currentFile)

stim = df.at[currentFile, 'Stim Prot']
ts=df.at[currentFile, 'Exp Time']
fs=1/ts 

percentile=98.6



###############################################
################### SIG PIXELS#################
###############################################

refoc_mean_stack=np.load(path + '\\stack_refoc\\refocused' + '\\' + 'refoc_mean_stack.npy')
reFstackA=refoc_mean_stack[:,round(len(refoc_mean_stack[1])/2),...]
darkTrialAverage_ref =90

back_ref=np.mean(reFstackA[:13,...],0)
df_ref=100*(reFstackA-back_ref[None,...]) / (back_ref[None,...] - darkTrialAverage_ref)

varImage_ref = np.var(df_ref,axis=-0)
# INTRA
#plt.imshow(varImage_ref)
x1=30
x2=70
y1=45
y2=85
y1=40
y2=80

varImageROI_ref = varImage_ref[y1:y2,x1:x2]
df_refROI = df_ref[:,y1:y2,x1:x2]
plt.imshow(varImageROI_ref)

signalPixels_ref= np.array(np.where(varImageROI_ref > np.percentile(varImageROI_ref,percentile)))


binarized = 1.0 * (varImageROI_ref > np.percentile(varImageROI_ref,percentile))
outlineROI=to_outline(binarized)

xS=50
lengthSB=25
pixelSize=5
y=56

fig,ax = pf.addROI(varImageROI_ref,outlineROI)
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
pf.noBorders(ax)
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
#pf.saveFigure(fig,path + r'\figures','dfImageROI-tv_{}per_{}it_25-SBrl'.format(percentile,it))


###############################################
#################### decon# ###################
###############################################

decon_RL_660=np.load(path + '\\stack_refoc\\deconvolved' + '\\' + '{}_decon_2201_intra_RL_full.npy'.format(currentFile))

num_iterations=[1,3,5,7,9,13,17,21]
darkTrialAverage_dec = 0


# 1 depth
decStackA= decon_RL_660[1,41,...]
back_dec=np.mean(decStackA[:13,...],0)
df_dec=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)
varImage_dec = np.var(df_dec,axis=0)


#all depths
df_dec=np.zeros((82,122,101,101))
for depth in range(len(decon_RL_660[1])):
    decStackA=decon_RL_660[1,depth,...]
    back_dec=np.mean(decStackA[:13,...],0)
    df_dec[depth]=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)
varImage_dec = np.var(df_dec,axis=1)              
               



varImageROI_dec = varImage_dec[:,y1:y2,x1:x2]
%varexp --imshow varImageROI_dec
df_decROI = df_dec[:,:,y1:y2,x1:x2]
signalPixels_dec = np.array(np.where(varImageROI_dec > np.percentile(varImageROI_dec,percentile)))


binarized = 1.0 * (varImageROI_dec > np.percentile(varImageROI_dec,percentile))
outlineROI=to_outline(binarized)

fig,ax = pf.addROI(varImageROI_dec,outlineROI)
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  

peakIdx=17

trialData_ref = np.average(df_refROI[:,signalPixels_ref[0],signalPixels_ref[1]], axis=1)
#trialData_dec = np.average(df_refROI[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)

# Ref pixels       
baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(peakIdx,trialData_ref,trialData_ref,darkTrialAverage_ref,13)
print(SNR,peak_dF_F, df_noise)

fields=['Ref','-',SNR, peak_dF_F, df_noise, bleach, percentile,path]
gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_wProcess_{}_660nmPSF_SPref_Jan22'.format(date),fields)

# Dec pixels   
#baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_dec,trialData_dec,darkTrialAverage_ref,13)
#print(SNR)
#
#fields=['Ref','-',SNR, peak_dF_F, df_noise, bleach, percentile,path]
#gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_wProcess_{}_660nmPSF_SPdec_Jan'.format(date),fields)
#



for ii in range(len(num_iterations)):
    print(num_iterations[ii])
    decStackA= decon_RL_660[ii,41,...]
    
    back_dec=np.mean(decStackA[:13,...],0)
    df_dec=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)
    
    varImage_dec = np.var(df_dec,axis=0)

    varImageROI_dec = varImage_dec[y1:y2,x1:x2]
    df_decROI = df_dec[:,y1:y2,x1:x2]
        
    trialData_ref = np.average(df_decROI[:,signalPixels_ref[0],signalPixels_ref[1]], axis=1)
#    trialData_dec = np.average(df_decROI[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)
    
    # Ref pixels       
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(peakIdx,trialData_ref,trialData_ref,darkTrialAverage_dec,13)
    print(SNR)
    
    fields=['Dec',num_iterations[ii],SNR, peak_dF_F, df_noise, bleach, percentile,path]
    gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_wProcess_{}_660nmPSF_SPref_Jan22'.format(date),fields)
    
    # Dec pixels   
#    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_dec,trialData_dec,darkTrialAverage_dec,13)
#    print(SNR)
#    
#    fields=['Dec',num_iterations[ii],SNR, peak_dF_F, df_noise, bleach, percentile,path]
#    gf.appendCSV(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\{}'.format(date),r'\stats_numIt_wProcess_{}_660nmPSF_SPdec_2'.format(date),fields)
#    
#    
    

    
xS=7.5
xE=9.5
yS=10
yE=20
    

time = np.arange(0, ts*len(trialData_ref), ts)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(time,average_wf+60,linewidth=3.0,color='k')
plt.plot(time,trialData_ref+30,linewidth=3.0,color='k')
plt.plot(time,trialData_dec,linewidth=3.0,color='k')

plt.plot([xS,xS],[yS,yE],linewidth=4.0,color='k')
plt.plot([xS,xE],[yS,yS],linewidth=4.0,color='k')

plt.plot([0.8,0.8],[68,70],linewidth=3.0,color='r')
plt.plot([1.8,1.8],[68,70],linewidth=3.0,color='r')
plt.plot([2.8,2.8],[68,70],linewidth=3.0,color='r')
plt.plot([3.8,3.8],[68,70],linewidth=3.0,color='r')
plt.plot([4.8,4.8],[68,70],linewidth=3.0,color='r')

fig.set_size_inches(8,8)

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.saveFigure(fig,path + r'\figures','timeSeries_wf_ref_dec-3it_SPref_SBs-10per_2sec')