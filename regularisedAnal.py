# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 07:48:12 2020

@author: chowe7
"""

# analyse regularised data
%reset


import scipy.ndimage as ndimage
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np
import sys
import pandas as pd
import imagingAnalysis as ia
import idx_refocus as ref
import deconvolveLF as dlf
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import plotting_functions as pf

def to_outline(roi):
    return np.logical_xor(ndimage.morphology.binary_dilation(roi),roi)

font = {'family': 'sans',
        'weight': 'normal',
        'size': 21,}

plt.rc('font',**font)

###############################################
################### INPUTS  ###################
###############################################
date = '190605'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra' + '\\' +date
currentFile = 'ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4' 
num_iterations=[1,3,5,7,9,13,17,21]
#regType=['RL','0.001','0.01']
regType=['RL','0.1','1.0']
percentile = 90

#indexes for figures. if first cell in file then diff =0 if second then diff=24 etc
idx=[0,7,8,15,16,23]
diff=0

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
path = cwd + sliceNo + cellNo + trialFolder + r'\regularised_decon_higher'

darkTrialData_dec=0
darkTrialAverage_dec =0
keyword='deconvolved'

figurePath =path + r'\figures'
gf.makeFolder(figurePath)

lengthSB = 25
pixelSize = 5

###############################################
################### load files ###################
###############################################
decon_reg_RL_full=np.load(path + '\\' + currentFile+ '_MMStack_Pos0.ome_high_carmel_decon_20201202_reg_RL_full.npy')
decon_reg_RL_reg_full=np.load(path +'\\' + currentFile+ '_MMStack_Pos0.ome_high_carmel_decon_20201202_reg_RL_reg_full.npy')
decon_reg_RL_reg2_full=np.load(path + '\\' + currentFile+ '_MMStack_Pos0.ome_high_carmel_decon_20201202_reg_RL_reg3_full.npy')


# background variance
varReg=np.zeros((len(num_iterations),3))
for it in range(len(decon_reg_RL_full)):
    varReg[it,0]=np.var(decon_reg_RL_full[it,round(len(decon_reg_RL_full[1])/2),10:30,10:30])
    varReg[it,1]=np.var(decon_reg_RL_reg_full[it,round(len(decon_reg_RL_full[1])/2),10:30,10:30])
    varReg[it,2]=np.var(decon_reg_RL_reg2_full[it,round(len(decon_reg_RL_full[1])/2),10:30,10:30])

fig = plt.figure()    
ax1 = plt.subplot(111)
plt.plot(num_iterations,varReg[:,0],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(num_iterations,varReg[:,1],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(num_iterations,varReg[:,2],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
plt.legend(frameon=False)
pf.lrBorders(ax1)
plt.ylabel('Variation', fontdict = font)
plt.xlabel('Iterations', fontdict = font)
plt.tight_layout() 
pf.saveFigure(fig,figurePath,'\\backgoundVariance')


# take the 3rd iteration all pixels at the native focal plane to get the signal pixels
init = decon_reg_RL_full[1,round(len(decon_reg_RL_full[1])/2),...]
#init = decon_reg_RL_reg2_full[1,round(len(decon_reg_RL_full[1])/2),...]

backgroundData_dec=np.average(init[:,10:30,10:30],axis=1)
backgroundData_dec=np.average(backgroundData_dec,axis=1)

back_dec=np.mean(init[:13,...],0)
df_dec=100*(init-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)

varImage_dec = np.var(df_dec,axis=-0)
plt.imshow(varImage_dec)

x1=28
x2=68
y1=43
y2=83
#x1=32
#x2=72
#y1=35
#y2=75
varImageROI_dec = varImage_dec[y1:y2,x1:x2]
plt.imshow(varImageROI_dec)

df_decROI = df_dec[:,y1:y2,x1:x2]
signalPixels_dec= np.array(np.where(varImageROI_dec > np.percentile(varImageROI_dec,percentile)))

binarized = 1.0 * (varImageROI_dec > np.percentile(varImageROI_dec,percentile))
outlineROI=to_outline(binarized)

y=37
xS=32

fig,ax = pf.addROI(varImageROI_dec,outlineROI)
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
pf.noBorders(ax)
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
pf.saveFigure(fig,figurePath,'dfImageROI_{}per_3it_25-SBrl'.format(percentile))
    
#save signal pixels 
np.save(path+r'\signalPixels_dec_{}per.npy'.format(percentile),signalPixels_dec)

#save images


# take all iterations all pixels at the native focal plane
decStackA=np.zeros((3,8,len(decon_reg_RL_full[0,0,:,...]),101,101))
decStackA[0] = decon_reg_RL_full[:,round(len(decon_reg_RL_full[1])/2),...]
decStackA[1] = decon_reg_RL_reg_full[:,round(len(decon_reg_RL_full[1])/2),...]
decStackA[2] = decon_reg_RL_reg2_full[:,round(len(decon_reg_RL_full[1])/2),...]
 
df_dec=np.zeros((3,8,len(decon_reg_RL_full[0,0,:,...]),101,101))
trialData_dec=np.zeros((3,8,len(decon_reg_RL_full[0,0,:,...])))

for ii in range(len(decStackA)):
    data=decStackA[ii,...]
    
    for it in range(len(data)):
        data_it = data[it]
        
        backgroundData_dec=np.average(data_it[:,10:30,10:30],axis=1)
        backgroundData_dec=np.average(backgroundData_dec,axis=1)

        back_dec=np.mean(data_it[:13,...],0)
        df_dec[ii,it]=100*(data_it-back_dec[None,...]) / (back_dec[None,...] - darkTrialAverage_dec)

        df_decROI = df_dec[ii,it,:,y1:y2,x1:x2]

        trialData_dec[ii,it] = np.average(df_decROI[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)

        # get stats
        baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach = ia.getStatistics(trialData_dec[ii,it],trialData_dec[ii,it],darkTrialData_dec,13)

        # save to excel
        fields=[currentFile,df.at[currentFile, 'slice'], df.at[currentFile, 'cell'],regType[ii],num_iterations[it],percentile,SNR,baseline, baselineNoise, peakSignal, peak_dF_F, df_noise, bleach]
        gf.appendCSV(cwd ,r'\stats_deconvolved_reg_{}'.format(date),fields)


# plots   
RL = trialData_dec[0,...]
reg=trialData_dec[1,...]
reg2=trialData_dec[2,...]

fig, axs = plt.subplots(3)
axs[0].plot(np.transpose(RL))
axs[1].plot(np.transpose(reg))
axs[2].plot(np.transpose(reg2))

fig, axs = plt.subplots(8)
for its in range(8):
    axs[its].plot(np.transpose(trialData_dec[:,its,:]))





#pd_ref = pd.read_csv(cwd + r'\\percentile\\stats_ref_percentile_{}.csv'.format(date),index_col = 'percentile')        
pd_dec = pd.read_csv(cwd + r'\stats_deconvolved_reg_{}.csv'.format(date))        


fig = plt.figure(figsize=(15, 5))    
ax1 = plt.subplot(131)
#plt.plot(pd_ref['peak F'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['peakF'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['peakF'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['peakF'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
plt.legend(frameon=False)
pf.lrBorders(ax1)
plt.ylabel('Peak Signal (%)', fontdict = font)

ax2 = plt.subplot(132)
#plt.plot(pd_ref['df noise'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['dfNoise'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['dfNoise'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['dfNoise'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax2)
plt.ylabel('Noise (%)', fontdict = font)
plt.xlabel('Iteration', fontdict = font)

ax3 = plt.subplot(133)
#plt.plot(pd_ref['SNR'],linewidth=3.0,color='k',label='Refocused')
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['SNR'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['SNR'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['SNR'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax3)
plt.ylabel('SNR (%)', fontdict = font)
plt.tight_layout() 
pf.saveFigure(fig,figurePath,'\\tutto')

#differences?
RL_reg2=decon_reg_RL_full[0,41,...] - decon_reg_RL_reg2_full[0,41,...]
RL_reg=decon_reg_RL_full[0,41,...] - decon_reg_RL_reg_full[0,41,...]
reg_reg2=decon_reg_RL_reg_full[0,41,...] - decon_reg_RL_reg2_full[0,41,...]




################################
#     Lateral Resolution       #
################################
#off static image

#currentFile = 'MLA_NOPIP_1x1_f_2-8_660nm_200mA_1' 
#currentFile = 'MLA3_1x1_50ms_150pA_1'
currentFile = 'MLA_1x1_50ms_150pA_1'
path = cwd + sliceNo + cellNo + r'\zstack' + '\\' + currentFile + r'\regularised_decon_new'
analysisFolder = path + '\\analysis'
gf.makeFolder(analysisFolder)
figurePath = path + '\\figures'
gf.makeFolder(figurePath)


decon_reg_RL_full=np.load(path + '\\' + currentFile+ '_MMStack_Pos0.ome_carmel_decon_20201123_reg_RL_full.npy')
decon_reg_RL_reg_full=np.load(path +'\\' + currentFile+ '_MMStack_Pos0.ome_carmel_decon_20201123_reg_RL_reg_full.npy')
decon_reg_RL_reg2_full=np.load(path + '\\' + currentFile+ '_MMStack_Pos0.ome_carmel_decon_20201123_reg_RL_reg3_full.npy')

pg.image(decon_reg_RL_full[0,41,...])

#inFocus=48
inFocus=15

decStackA=np.zeros((3,8,82,101,101))
decStackA[0] = decon_reg_RL_full[:,:,inFocus,...]
decStackA[1] = decon_reg_RL_reg_full[:,:,inFocus,...]
decStackA[2] = decon_reg_RL_reg2_full[:,:,inFocus,...]

xLat=np.linspace(0, decStackA.shape[4]*pixelSize, decStackA.shape[4])  #x in um
xAx=np.linspace(0, decStackA.shape[2], decStackA.shape[2])  #x in um

################################
#     Lateral Resolution       #
################################
# xy and yx are the 2d images at the native focal plane
#got to manually set the x or y location of the cell
plt.imshow(decStackA[0,0,41,:,:])

#xLoc=39
#yLoc=58
xLoc=50
yLoc=50
xLoc=51
yLoc=58

xy=np.zeros((3,8,101,1))
yx=np.zeros((3,8,1,101))

for ii in range(3):
    deconvolved_result = decStackA[ii,...]
    
    for it in range(8):
        data_it = deconvolved_result[it]
        
        xy[ii,it] = data_it[41,:,xLoc:xLoc+1]
        yx[ii,it] = data_it[41,yLoc:yLoc+1,:]
        

FWHM_xy=np.zeros((3,8,1))
FWHM_yx=np.zeros((3,8,1))
for ii in range(len(xy)):     
    for it in range(8):   
    
        print(num_iterations[it])
        y=xy[ii,it,:,0]
        spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
        r1, r2 = spline.roots() # find the roots
        FWHM_xy=r2-r1
        
        y=yx[ii,it,0,:]
        spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
        r1, r2 = spline.roots() # find the roots
        FWHM_yx=r2-r1
        
        fields=[date,regType[ii],num_iterations[it],FWHM_xy,FWHM_yx,xLoc,yLoc]
        gf.appendCSV(analysisFolder ,r'\lateralResolution_deconReg_FWHM',fields)


xz=np.zeros((3,8,82,101))
yz=np.zeros((3,8,82,101))

for ii in range(3):
    deconvolved_result = decStackA[ii,...]
    
    for it in range(8):
        data_it = deconvolved_result[it]
        
        xz[ii,it]=data_it[:,:,xLoc]
        yz[ii,it]=data_it[:,yLoc,:]
        
        
for ii in range(len(xy)):     
    for it in range(8):   
    
        print(num_iterations[it])
        y=xz[ii,it,:,yLoc]    
        try:
            spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
            r1, r2 = spline.roots() # find the roots
        except: # fit a gaussian 
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
        
        y=yz[ii,it,:,xLoc]  
        try:
            spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
            r1, r2 = spline.roots() # find the roots
        except: 
        # fit a gaussian 
            gaussian = lambda x: 3*np.exp(-(30-x)**2/20.)
            xAx = np.arange(y.size)
            x = np.sum(xAx*y)/np.sum(y)
            width = np.sqrt(np.abs(np.sum((xAx-x)**2*y)/np.sum(y)))
            max = y.max()
            fit = lambda t : max*np.exp(-(t-x)**2/(1*width**2))
            yFit = fit(xAx)
            spline = UnivariateSpline(xAx, yFit-np.max(yFit)/2, s=0)
            r1, r2 = spline.roots() # find the roots
        FWHM_yz=r2-r1
        print(FWHM_yz)
        fields=[date,regType[ii],num_iterations[it],FWHM_xz,FWHM_yz,xLoc,yLoc]
        gf.appendCSV(analysisFolder ,r'\axialResolution_deconReg_FWHM',fields)

#40:70
for ii in range(len(xy)): 
    for it in range(8):   
        print(num_iterations[it])
        fig = plt.gcf()      
        ax = plt.subplot(111)  
        plt.imshow(xz[ii,it,20:60,45:75])  
        
        #axis formatting
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(False)
        ax.spines['left'].set_linewidth(False)
        ax.axes.get_yaxis().set_ticks([])
        ax.axes.get_xaxis().set_ticks([]) 
        plt.tight_layout()  
        plt.savefig(figurePath + r'\\xz_deconReg_{}_it-{}_z-40um.png'.format(regType[ii],num_iterations[it]), format='png', dpi=600, bbox_inches='tight')
        plt.close(fig)
        
        
color_idx = np.linspace(0, 1, 9)

fig = plt.gcf()    
ax = fig.add_subplot(131)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xAx[20:60]-41,xz[0,ii,20:60,yLoc]*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.ylabel('e-3', fontdict = font)
    plt.ylim([-0.01,0.5])
ax = fig.add_subplot(132)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xAx[20:60]-41,xz[1,ii,20:60,yLoc]*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.xlabel('Depth (um)', fontdict = font)
    plt.ylim([-0.01,0.5])
ax = fig.add_subplot(133)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xAx[20:60]-41,xz[2,ii,20:60,yLoc]*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    plt.ylim([-0.01,0.5])
fig.set_size_inches(14,6)
plt.tight_layout()  
plt.savefig(figurePath + r'\\xzLine_deconReg.png', format='png', dpi=600, bbox_inches='tight')
plt.close(fig)   
    

         

for ii in range(len(xy)):     
    for it in range(8):   
        print(num_iterations[it])
        fig = plt.gcf()      
        ax = plt.subplot(111)  
        plt.imshow(decStackA[ii,it,41,xLoc-20:xLoc+20,yLoc-20:yLoc+20])  
        pf.noBorders(ax)
        plt.tight_layout()  
        plt.savefig(figurePath + r'\\FOVxy_deconReg_{}_it-{}.png'.format(regType[ii],num_iterations[it]), format='png', dpi=600, bbox_inches='tight')
        plt.close(fig)


fig = plt.gcf()    
ax1 = fig.add_subplot(131)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xLat[yLoc-20:yLoc+20],xy[0,ii,yLoc-20:yLoc+20,:]*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
    pf.lrBorders(ax1)
    plt.ylabel('e-3', fontdict = font)
    plt.ylim([-0.01,0.5])
ax2 = fig.add_subplot(132)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xLat[yLoc-20:yLoc+20],xy[1,ii,yLoc-20:yLoc+20,:]*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    pf.lrBorders(ax2)
    plt.xlabel('xy (um)', fontdict = font)
    plt.ylim([-0.01,0.5])
ax3 = fig.add_subplot(133)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xLat[yLoc-20:yLoc+20],xy[2,ii,yLoc-20:yLoc+20,:]*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    pf.lrBorders(ax3)
    plt.ylim([-0.01,0.5])
fig.set_size_inches(14,6)
plt.tight_layout()  
plt.savefig(figurePath + r'\\xyLine_deconReg.png', format='png', dpi=600, bbox_inches='tight')
plt.close(fig) 



fig = plt.gcf()    
ax1 = fig.add_subplot(131)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xLat[xLoc-20:xLoc+20],np.transpose(yx[0,ii,:,xLoc-20:xLoc+20])*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
    pf.lrBorders(ax1)
    plt.ylabel('e-3', fontdict = font)
    plt.ylim([-0.01,0.55])
ax2 = fig.add_subplot(132)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xLat[xLoc-20:xLoc+20],np.transpose(yx[1,ii,:,xLoc-20:xLoc+20])*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    pf.lrBorders(ax2)
    plt.xlabel('yx (um)', fontdict = font)
    plt.ylim([-0.01,0.55])
ax3 = fig.add_subplot(133)
for col,ii in zip(color_idx, range(8)):
    lines=plt.plot(xLat[xLoc-20:xLoc+20],np.transpose(yx[2,ii,:,xLoc-20:xLoc+20])*1000,linewidth=2.5, color=plt.cm.plasma(col))    
    pf.lrBorders(ax3)
    plt.ylim([-0.01,0.55])
fig.set_size_inches(14,6)
plt.tight_layout()  
plt.savefig(figurePath + r'\\yxLine_deconReg.png', format='png', dpi=600, bbox_inches='tight')
plt.close(fig) 

################################
#       Summary Figures        #
################################

pd_latRes = pd.read_csv(analysisFolder + r'\lateralResolution_deconReg_FWHM.csv')  
pd_axRes = pd.read_csv(analysisFolder + r'\axialResolution_deconReg_FWHM.csv')  

fig = plt.figure()    
ax1 = plt.subplot(221)
plt.plot(pd_latRes['numIt'][0:7],pd_latRes['xy'][0:7],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_latRes['numIt'][8:15],pd_latRes['xy'][8:15],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_latRes['numIt'][16:23],pd_latRes['xy'][16:23],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax1)
plt.ylabel('xy (um)', fontdict = font)

ax2 = plt.subplot(222)
plt.plot(pd_latRes['numIt'][0:7],pd_latRes['yx'][0:7],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_latRes['numIt'][8:15],pd_latRes['yx'][8:15],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_latRes['numIt'][16:23],pd_latRes['yx'][16:23],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
plt.legend(frameon=False)
pf.lrBorders(ax2)
plt.ylabel('yx (um)', fontdict = font)

ax3 = plt.subplot(223)
plt.plot(pd_axRes['numIt'][0:7],pd_axRes['xz'][0:7],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_axRes['numIt'][8:15],pd_axRes['xz'][8:15],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_axRes['numIt'][16:23],pd_axRes['xz'][16:23],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax3)
plt.ylabel('xz (um)', fontdict = font)
plt.xlabel('Iterations', fontdict = font)

ax4 = plt.subplot(224)
plt.plot(pd_axRes['numIt'][0:7],pd_axRes['yz'][0:7],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_axRes['numIt'][8:15],pd_axRes['yz'][8:15],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_axRes['numIt'][16:23],pd_axRes['yz'][16:23],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax4)
plt.ylabel('yz (um)', fontdict = font)

fig.set_size_inches(10,8)
plt.tight_layout()  
plt.savefig(figurePath + r'\\spatialSummary_deconReg.png', format='png', dpi=600, bbox_inches='tight')
plt.close(fig) 