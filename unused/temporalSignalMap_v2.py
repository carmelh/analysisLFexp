# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:21:29 2020

@author: chowe7
"""

#3d temporal map
import pyqtgraph as pg
import numpy as np
import sys
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import plotting_functions as pf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

keyword='deconvolved'
path = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4'
pathDarkTrial = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM-DARK_1'
path_stack = path + '\stack_refoc' + '\\' + keyword
figureFolder = path + '\\figures\\3D'
stim = 'A Stim'
num_iterations=3


percentile=98

#ref dark
darkTrialData_ref = 90

#dec dark
darkTrialData_dec = 0


## get files 
## get in focus for signal pixels
reFstackA=np.array(refoc_mean_stack[:,40,...] )

back_ref=np.mean(reFstackA[:13,...],0)
df_ref=100*(reFstackA-back_ref[None,...]) / (back_ref[None,...] - darkTrialData_ref)

varImage_ref = np.var(df_ref,axis=-0)
#plt.imshow(varImage_ref)
x1=25
x2=75
y1=40
y2=90
varImageROI_ref = varImage_ref[y1:y2,x1:x2]
df_refROI = df_ref[:,y1:y2,x1:x2]
plt.imshow(varImageROI_ref)

signalPixels_ref= np.array(np.where(varImageROI_ref > np.percentile(varImageROI_ref,percentile)))


#for static fov images
lengthSB = 25
pixelSize = 5
xS = 30
y = 36


#################
#  DECONVOLVED  #
#################

decon_mean_stack= decon_mean_stack_it[1,...]

decon_mean_stack= decon_mean_stack_660_RL_it3[:,:,y1:y2,x1:x2]


back_dec=np.mean(decon_mean_stack[:13,...],0)
df_dec=100*(decon_mean_stack-back_dec[None,...]) / (back_dec[None,...] - darkTrialData_dec)

# iterate through depth 
processedTracesD_soma = np.zeros((len(decon_mean_stack),len(decon_mean_stack[0])))
processedTracesD_process1 = np.zeros((len(decon_mean_stack),len(decon_mean_stack[0])))
processedTracesD_process2 = np.zeros((len(decon_mean_stack[0]),len(decon_mean_stack)))
processedTracesD_process3 = np.zeros((len(decon_mean_stack[0]),len(decon_mean_stack)))
varImages_dec = np.zeros((len(decon_mean_stack[1]),101,101))


back_dec=np.mean(decon_mean_stack[:,:13,...],1)
df_dec=100*(decon_mean_stack-back_dec[None,...]) / (back_dec[None,...] - darkTrialData_dec)


df_dec = np.zeros((len(decon_mean_stack),len(decon_mean_stack[1]),101,101))


for z in range(len(decon_mean_stack)):
    print(z)
    decStackA= decon_mean_stack[z,...]
    
    back_dec=np.mean(decStackA[:13,...],0)
    df_dec[z]=100*(decStackA-back_dec[None,...]) / (back_dec[None,...] - darkTrialData_dec)
    

for z in range(len(df_dec)):
    trialData_p=np.average(df_dec[z,:,48:50,45:46],axis=1)
    processedTracesD_process1[z]=np.average(trialData_p,axis=1)  
    
    processedTracesD_soma = np.average(df_dec[z,:,63:66,46:48], axis=1)
    processedTracesD_soma[z]=np.average(processedTracesD_soma,axis=1)
    
    trialData_p=np.average(df_dec[z,:,54:58,39:40],axis=1)
    processedTracesD_process2[z]=np.average(trialData_p,axis=1)  
    
    trialData_p=np.average(df_dec[z,:,70:72,56:60],axis=1)
    processedTracesD_process3[z]=np.average(trialData_p,axis=1)  

#   stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTracesD_soma[z],processedTracesD_soma[z],darkTrialData_dec,13)

    if z == 0:
        fields=['path','depth','SNR','baseline', 'baseline_photons',' baselineNoise', 'peakSignal', 'peakSignal_photons', 'peak_dF_F', 'df_noise', 'bleach', 'baselineDarkNoise']
#        gf.appendCSV(path ,r'\depthDeconSTATS',fields)
        gf.appendCSV(path ,r'\depthDeconSTATS_process_ROI-3',fields)
        
#    fields=[path,z,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, baselineDarkNoise]
#    gf.appendCSV(path ,r'\depthDeconSTATS',fields)

#   process stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTracesD_process3[z],processedTracesD_process3[z],darkTrialData_dec,13)

    fields=[path,z,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, baselineDarkNoise]
    gf.appendCSV(path ,r'\depthDeconSTATS_process_ROI-3',fields)
    print('Finished Processing depth {}'.format(z))
    

pf.plotImage(df[17,40,42:82,27:67])
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
plt.savefig(figureFolder + r'\\image_deconvolved.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\image_deconvolved.eps', format='eps', dpi=800, bbox_inches='tight')



#fig = plt.gcf()    
#ax = fig.add_subplot(111)
#for col,ii in zip(color_idx, range(len(trialData_soma))):
#    lines=plt.plot(trialData_soma[ii],linewidth=2.5, color=plt.cm.plasma(col))
#    
    
###############
#  REFOCUSED  #
###############
    
processedTracesR_soma = np.zeros((len(refoc_mean_stack[0]),len(refoc_mean_stack)))
processedTracesR_process1 = np.zeros((len(refoc_mean_stack[0]),len(refoc_mean_stack)))
processedTracesR_process2 = np.zeros((len(refoc_mean_stack[0]),len(refoc_mean_stack)))
processedTracesR_process3 = np.zeros((len(refoc_mean_stack[0]),len(refoc_mean_stack)))
#varImages_ref = np.zeros((len(refoc_mean_stack[1]),101,101))

back_ref=np.mean(refoc_mean_stack[:13,...],0)
df_ref=100*(refoc_mean_stack-back_ref[None,...]) / (back_ref[None,...] - darkTrialData_ref)

for z in range(len(refoc_mean_stack[0])):
    trialData_p=np.average(df_ref[:,z,48:50,45:46],axis=1)
    processedTracesR_process1[z]=np.average(trialData_p,axis=1)  
    processedTracesR_soma[z] = np.average(df_ref[:,z,signalPixels[0],signalPixels[1]], axis=1) 
    trialData_p=np.average(df_ref[:,z,54:58,39:40],axis=1)
    processedTracesR_process2[z]=np.average(trialData_p,axis=1)  
    trialData_p=np.average(df_ref[:,z,70:72,56:60],axis=1)
    processedTracesR_process3[z]=np.average(trialData_p,axis=1)  
    
    
#   soma stats
#    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTracesR_soma[z],trialData_soma,darkTrialData_ref,baselineIdx)

    if z == 0:
        fields=['path','depth','SNR','baseline', 'baseline_photons',' baselineNoise', 'peakSignal', 'peakSignal_photons', 'peak_dF_F', 'df_noise', 'bleach', 'baselineDarkNoise']
#        gf.appendCSV(path ,r'\depthRefocusedSTATS',fields)
        gf.appendCSV(path ,r'\depthRefocusedSTATS_process_ROI-2022',fields)      
        
#    fields=[path,z,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, baselineDarkNoise]
#    gf.appendCSV(path ,r'\depthRefocusedSTATS',fields)
    
    #process stats
    baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(processedTracesR_process[z],trialData_p,darkTrialData_ref,baselineIdx)
        
    fields=[path,z,SNR,baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, bleach, baselineDarkNoise]
    gf.appendCSV(path ,r'\depthRefocusedSTATS_process_ROI-2022',fields) 
    print('Finished Processing depth {}'.format(z))
    

pf.plotImage(df[17,40,42:82,27:67])
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
plt.savefig(figureFolder + r'\\image_refocused.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\image_refocused.eps', format='eps', dpi=800, bbox_inches='tight')



#for z in range(42):
#    reFstackA =[]
#    for ii in range(122):
#        test = decon[:,ii,...]
#        reFstackA.append(test[z,:,:])
#
#    reFstackA=np.array(reFstackA)
#    backgroundData=np.average(reFstackA[:,10:30,10:30],axis=1)
#    backgroundData=np.average(backgroundData,axis=1)
#    trialData=np.average(reFstackA[:,57:58,43:44],axis=1)
#    trialData=np.average(trialData,axis=1)  
##    trialData = np.average(reFstackA[:,signalPixels[0][4:9],signalPixels[1][4:9]], axis=1)  
#    processedTraces[:,z], diffROI,processedBackgroundTrace,baselineIdx = ia.processRawTrace(trialData, darkTrialData, backgroundData, stim)
#    print('Finished Processing depth {}'.format(z))



keyword='refocused_soma_80um-1umSteps'
keyword='refocused_process_80um-1umSteps'
keyword='deconvolved_soma_80um-1umSteps'
keyword='deconvolved_process_80um-1umSteps'


fig = plt.gcf()      
ax = plt.subplot(111)  
im = plt.imshow(processedTracesD_soma[:,0:122])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.3)
plt.colorbar(im, cax=cax)
fig.set_size_inches(9,6)
plt.tight_layout() 
plt.savefig(path + r'\\figures\\3D\\timeSeries3D_{}.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(path + r'\\figures\\3D\\timeSeries3D_{}.eps'.format(keyword), format='eps', dpi=800, bbox_inches='tight')    






xS=270
lengthSB = 25
pixelSize = 0.26
y = 360

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(stack[0,round(52/0.05):round(72/0.05),round(37/0.05):round(57/0.05)])
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
plt.savefig(figureFolder + '\\image_WF_25umSB.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + '\\image_WF_25umSB.eps', format='eps', dpi=800, bbox_inches='tight')



####################################
#      Depth Summary Figures       #
####################################
    
#maxProj_dec = np.max(varImages_dec, axis=0)





##########
#  SOMA  #
##########

df_dec = pd.read_csv(path + '\\depthDeconSTATS.csv')
df_ref = pd.read_csv(path + '\\depthRefocusedSTATS.csv')

fig = plt.gcf()      
ax = plt.subplot(111)
plt.plot(df_ref['depth']-(len(df_ref['depth'])/2),df_ref['peak_dF_F'],color='k',linewidth='3',label='Refocused')
plt.plot(df_dec['depth']-(len(df_dec['depth'])/2),df_dec['peak_dF_F'],color='r',linewidth='3',label='Deconvolved')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(2)
#ax.set_ylim((2,24))
fig.set_size_inches(6,3)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\peakSig-depth_soma.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\peakSig-depth_soma.eps', format='eps', dpi=800, bbox_inches='tight')



#############
#  PROCESS  #
#############
depth = np.arange(-40.5,40.5,1)

df_dec = pd.read_csv(path + '\\depthDeconSTATS_process_ROI-2.csv')
df_ref = pd.read_csv(path + '\\depthRefocusedSTATS_process_ROI-2.csv')

fig = plt.gcf()      
ax = plt.subplot(111)
plt.plot(df_ref['depth']-(len(df_ref['depth'])/2),df_ref['peak_dF_F'],color='k',linewidth='3',label='Refocused')
plt.plot(df_dec['depth']-(len(df_dec['depth'])/2),df_dec['peak_dF_F'],color='r',linewidth='3',label='Deconvolved')
#plt.plot(depth,timeMax,color='b',linewidth='3',label='Decon. Max')
#plt.plot(depth,timeSum/10,color='g',linewidth='3',label='Decon. Sum')

#axis formatting
#plt.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(2)
#ax.set_ylim((2,24))
fig.set_size_inches(6,3)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\peakSig-depth_process-1.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\peakSig-depth_process-1.eps', format='eps', dpi=800, bbox_inches='tight')



#############
#  SUM  #
#############
depth = np.arange(-40,41,1)
depth_ref = np.arange(-40,40,1)


timeSum_dec = np.sum(processedTracesD_process3,axis=1)
timeSum_ref = np.sum(processedTracesR_process3,axis=1)

#_process1

fig = plt.gcf()      
ax = plt.subplot(111)
plt.plot(depth_ref,timeSum_ref,color='k',linewidth='3',label='Ref. Sum')
plt.plot(depth,timeSum_dec,color='r',linewidth='3',label='Decon. Sum')

#axis formatting
#plt.legend(frameon=False)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(2)
#ax.set_ylim((2,24))
fig.set_size_inches(6,3)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\sumSig-depth_process-ROI-3.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\sumSig-depth_process-ROI-3.eps', format='eps', dpi=800, bbox_inches='tight')


_process-ROI-1


gaussian = lambda x: 3*np.exp(-(30-x)**2/20.)
xAx = np.arange(y.size)
x = np.sum(xAx*y)/np.sum(y)
width = np.sqrt(np.abs(np.sum((xAx-x)**2*y)/np.sum(y)))
max = y.max()
fit = lambda t : max*np.exp(-(t-x)**2/(2*width**2))
yFit = fit(xAx)
spline = UnivariateSpline(xAx, yFit-np.max(yFit)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xz=r2-r1






y=timeSum_dec
spline = UnivariateSpline(depth, y-np.max(y)/2, s=0)
r1, r2, r3, r4 = spline.roots() # find the roots
FWHM=r2-r1
print(FWHM)


y=timeSum_ref
spline = UnivariateSpline(depth_ref, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM=r2-r1
print(FWHM)


fig = plt.gcf()      
ax = plt.subplot(111)  
pl.plot(depth_ref, y)
pl.axvspan(r1, r2, facecolor='g', alpha=0.5)