# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:40:40 2020

@author: chowe7
"""
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np
import sys
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import pandas as pd
import plotting_functions as pf


font = {'family': 'sans',
        'weight': 'normal',
        'size': 25,}

plt.rc('font',**font)

def to_outline(roi):
    return np.logical_xor(ndimage.morphology.binary_dilation(roi),roi)


###############################################
################### INPUTS  ###################
###############################################MLA2_1x1_50ms_200pA_A_Stim_1
date = '190605'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra' + '\\' +date
currentFile = 'ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4' 
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

decon_mean_stack=np.load(path + '\\stack_refoc\\deconvolved\\decon_mean_stack.npy')
refoc_mean_stack=np.load(path + '\\stack_refoc\\refocused\\refoc_mean_stack.npy')



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

################### deconvolved ###################

keyword='deconvolved'
varImage_dec = np.var(df_dec,axis=-0)

plt.imshow(varImage_dec)

x1=30
x2=70
y1=45
y2=85

varImageROI_dec = varImage_dec[y1:y2,x1:x2]
plt.imshow(varImageROI_dec)

df_decROI = df_dec[:,y1:y2,x1:x2]


################### deconvolved ###################

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

varImage_ref = np.var(df_ref,axis=-0)

varImageROI_ref = varImage_ref[y1:y2,x1:x2]
df_refROI = df_ref[:,y1:y2,x1:x2]


outlineROI = np.zeros((100,40,40))
percentile=np.arange(90,99.9,0.1)
for pp in range(len(percentile)):
    signalPixels_ref= np.array(np.where(varImageROI_ref > np.percentile(varImageROI_ref,percentile[pp])))
    np.save(path + r'\\percentile\\signalPixels\\' + 'signalPixels-{}.npy'.format(percentile[pp]),signalPixels_ref)

    trialData_ref = np.average(df_refROI[:,signalPixels_ref[0],signalPixels_ref[1]], axis=1)

    binarized = 1.0 * (varImageROI_ref > np.percentile(varImageROI_ref,percentile[pp]))
    outlineROI[pp]=gf.to_outline(binarized)

# get stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_ref,trialData_ref,darkTrialData_ref,13)

# save to excel
    fields=[percentile[pp],SNR, peak_dF_F, df_noise]
    gf.appendCSV(path + r'\\percentile\\', '\\stats_ref_percentile_{}'.format(date),fields)
       
    trialData_dec = np.average(df_decROI[:,signalPixels_ref[0],signalPixels_ref[1]], axis=1)

    # get stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_dec,trialData_dec,darkTrialData_dec,13)

    fields=[percentile[pp],SNR,peak_dF_F, df_noise]
    gf.appendCSV(path + r'\\percentile\\', '\\stats_dec_percentile_{}'.format(date),fields)
    
    

# summary figures

pd_ref = pd.read_csv(path + r'\\percentile\\stats_ref_percentile_{}.csv'.format(date),index_col = 'percentile')        
pd_dec = pd.read_csv(path + r'\\percentile\\stats_dec_percentile_{}.csv'.format(date),index_col = 'percentile')        

# tutto
fig = plt.gcf()      
ax1 = plt.subplot(131)
plt.plot(pd_ref['peak F'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['peak F'],linewidth=3.0,color='r',label='Deconvolved')
plt.legend(frameon=False)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
plt.ylabel('Peak Signal (%)', fontdict = font)

ax2 = plt.subplot(132)
plt.plot(pd_ref['df noise'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['df noise'],linewidth=3.0,color='r',label='Deconvolved')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.ylabel('Noise (%)', fontdict = font)
plt.xlabel('Percentile', fontdict = font)

ax3 = plt.subplot(133)
plt.plot(pd_ref['SNR'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['SNR'],linewidth=3.0,color='r',label='Deconvolved')
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_linewidth(2)
ax3.spines['left'].set_linewidth(2)
plt.ylabel('SNR (%)', fontdict = font)

fig.set_size_inches(20,7)
plt.tight_layout()
pf.saveFigure(fig,path + r'\\percentile\\','tutto')


#separate figures
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(pd_ref['peak F'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['peak F'],linewidth=3.0,color='r',label='Deconvolved')

#axis formatting
plt.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel('Percentile', fontdict = font)
plt.ylabel('Peak Signal (%)', fontdict = font)
fig.set_size_inches(7,5.5)
plt.tight_layout()
pf.saveFigure(fig,path + r'\\percentile\\','peakSignal')

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(pd_ref['df noise'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['df noise'],linewidth=3.0,color='r',label='Deconvolved')

#axis formatting
plt.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel('Percentile', fontdict = font)
plt.ylabel('Noise (%)', fontdict = font)
fig.set_size_inches(7,5.5)
plt.tight_layout()
pf.saveFigure(fig,path + r'\\percentile\\','noise')

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(pd_ref['SNR'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['SNR'],linewidth=3.0,color='r',label='Deconvolved')

#axis formatting
plt.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel('Percentile', fontdict = font)
plt.ylabel('SNR', fontdict = font)
fig.set_size_inches(7,5.5)
plt.tight_layout()
pf.saveFigure(fig,path + r'\\percentile\\','SNR')


# combo of all cells
