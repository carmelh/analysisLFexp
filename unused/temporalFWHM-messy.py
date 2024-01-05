# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:19:12 2020

@author: chowe7
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
from scipy.signal import find_peaks, peak_widths


def func_exp(x, a, b, c):
    return a * np.exp(-b * x) + c

def func_log(x, a, b, c):
  return a * np.log(b * x) + c

font = {'family': 'sans',
    'weight': 'normal',
    'size': 24,}

plt.rc('font',**font)

pathWF = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\NOMLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_1'
pathDarkWF = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\NOMLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM-NOLGHT_1'

x,trialData = gf.importCSV(pathWF,'\\NOMLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_1_MMStack_Pos0.ome')
x,backgroundData = gf.importCSV(pathWF,'\\NOMLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_1_MMStack_Pos0-BG.ome')
x,darkTrialData = gf.importCSV(pathDarkWF,'\\WF2_1x1_50ms_100pA_A_Stim_DARK_1_MMStack_Pos0.ome')


path = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4'
percentile=98

#dec
decStackA=gf.loadPickles(path,'\\deconvolvedStack_RL_infocus_3')
darkTrialData_dec = 0
back=np.mean(decStackA[:13,...],0)
df_dec=100*(decStackA-back[None,...]) / (back[None,...] - darkTrialData_dec)
varImage_dec = np.var(df_dec,axis=-0)
signalPixels_dec= np.array(np.where(varImage_dec > np.percentile(varImage_dec,percentile)))
binarized = 1.0 * (varImage_dec > np.percentile(varImage_dec,percentile))
outlineROI=to_outline(binarized)
trialDataDec_process = np.average(df_dec[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)   


#ref
reFstackA=gf.loadPickles(path,'\\refstack_infocus')    
darkTrialData_ref = 90
back=np.mean(reFstackA[:13,...],0)
df_ref=100*(reFstackA-back[None,...]) / (back[None,...] - darkTrialData_ref)
varImage_ref = np.var(df_ref,axis=-0)
trialDataRef_process = np.average(df_ref[:,signalPixels_dec[0],signalPixels_dec[1]], axis=1)   


varImage_ref2 = np.var(reFstackA,axis=-0)
signalPixels_soma= np.array(np.where(varImage_ref2 > np.percentile(varImage_ref2,99.92)))
binarized_soma = 1.0 * (varImage_ref2 > np.percentile(varImage_ref2,99.94))
outlineROI_soma=to_outline(binarized_soma)
trialDataDec_soma= np.average(df_dec[:,signalPixels_soma[0],signalPixels_soma[1]], axis=1)  
trialDataRef_soma = np.average(df_ref[:,signalPixels_soma[0],signalPixels_soma[1]], axis=1) 

plt.figure()
plt.plot(trialDataDec_process)
plt.plot(trialDataRef_process)


avg2=(trialDataDec_soma_2 + trialDataDec_soma)/2
plt.figure()
plt.plot(trialDataDec_soma_2)
plt.plot(trialDataDec_soma)
plt.plot(test)


plt.figure()
plt.plot(trialDataDec_soma_2)
plt.plot(trialDataRef_soma_2)
plt.plot(trialDataDec_soma)
plt.plot(trialDataRef_soma)



processedTrace=trialDataDec_soma
processedTrace_2=trialDataDec_soma_2

processedTrace=trialDataRef_soma
processedTrace_2=trialDataRef_soma_2

processedTrace=trialDataDec_process
processedTrace_2=trialDataDec_process_2

processedTrace=trialDataRef_process
processedTrace_2=trialDataRef_process_2

yn0=np.array(processedTrace[16:36])+np.min(processedTrace[16:36])*-1
yn1=np.array(processedTrace[36:56])+np.min(processedTrace[36:56])*-1
yn2=np.array(processedTrace[56:76])+np.min(processedTrace[56:76])*-1
yn3=np.array(processedTrace[76:96])+np.min(processedTrace[76:96])*-1
yn4=np.array(processedTrace[96:116])+np.min(processedTrace[96:116])*-1
yn5=np.array(processedTrace_2[16:36])+np.min(processedTrace_2[16:36])*-1
yn6=np.array(processedTrace_2[36:56])+np.min(processedTrace_2[36:56])*-1
yn7=np.array(processedTrace_2[56:76])+np.min(processedTrace_2[56:76])*-1
yn8=np.array(processedTrace_2[76:96])+np.min(processedTrace_2[76:96])*-1
yn9=np.array(processedTrace_2[96:116])+np.min(processedTrace_2[96:116])*-1


x= np.linspace(0,len(yn0)*0.05,len(yn0))

y=yn0+yn1+yn2+yn3+yn4+yn5+yn6+yn7+yn8+yn9/10



spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots

FWHM_refSoma=r2-r1
print(FWHM_refSoma)

FWHM_refProcess=r2-r1
print(FWHM_refProcess)

FWHM_decSoma=r2-r1
print(FWHM_decSoma)

FWHM_decProcess=r2-r1
print(FWHM_decProcess)


currentFile = '190605\slice1\Cell2'
resultsFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\TemporalFWHM'

fields=[currentFile,FWHM_refSoma,FWHM_refProcess,FWHM_decSoma,FWHM_decProcess,'10']
gf.appendCSV(resultsFolder ,r'\temporalFWHM',fields)



fig = plt.gcf()      
ax = plt.subplot(111)  
#plt.plot(x,gf.norm(processedTrace_wf),linewidth=3.0,color='k',label='Widefield')
plt.plot(x,gf.norm(y_refSoma),linewidth=3.0,color='k',label='Refocused')
plt.plot(x,gf.norm(y_refP),linestyle='dashed',linewidth=3.0,color='k',label='Deconvolved')
plt.plot(x,gf.norm(y_decSoma),linewidth=3.0,color='r',label='Deconvolved')
plt.plot(x,gf.norm(y_decP),linestyle='dashed',linewidth=3.0,color='r',label='Refocused')
plt.plot([0,1],[-0.05,-0.05],linewidth=4.0,color='k') 
fig.set_size_inches(10,4)
#plt.legend(loc='upper left',frameon=False)

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.savefig(resultsFolder + r'\\temporalAverage_190605.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(resultsFolder + r'\\temporalAverage_190605.eps', format='eps', dpi=800, bbox_inches='tight')
plt.close(fig)  









# FWHM 
xn0 = np.arange(17,35,1)
xn1 = np.arange(37,55,1)
xn2 = np.arange(57,75,1)
xn3 = np.arange(77,95,1)
xn4 = np.arange(97,115,1)

yn0=np.array(processedTrace[17:35])+1
yn1=np.array(processedTrace[37:55])+1
yn2=np.array(processedTrace[57:75])+1
yn3=np.array(processedTrace[77:95])+1
yn4=np.array(processedTrace[97:115])+1

popt0, pcov0 = curve_fit(func_exp, xn0, yn0,p0=[40,0.4,1])
popt1, pcov1 = curve_fit(func_exp, xn1, yn1,p0=[88,0.4,1])
popt2, pcov2 = curve_fit(func_exp, xn2, yn2,p0=[8e5,0.4,1])
popt3, pcov3 = curve_fit(func_exp, xn3, yn3,p0=[8e10,0.4,1])
popt4, pcov4 = curve_fit(func_exp, xn4, yn4,p0=[8e10,0.4,1])

#refocused
#popt1, pcov1 = curve_fit(func_exp, xn1, yn1,p0=[0.02,0.08,1])
#popt2, pcov2 = curve_fit(func_exp, xn2, yn2,p0=[0.02,0.08,1])
#popt3, pcov3 = curve_fit(func_exp, xn3, yn3,p0=[5,0.08,1])
#popt4, pcov4 = curve_fit(func_exp, xn4, yn4,p0=[100,1,1])

plt.figure()
plt.plot(xn0, yn0, 'ko', label="Original Noised Data")
plt.plot(xn0, func_exp(xn0, *popt0), 'r-', label="Fitted Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(xn1, yn1, 'ko', label="Original Noised Data")
plt.plot(xn1, func_exp(xn1, *popt1), 'r-', label="Fitted Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(xn2, yn2, 'ko', label="Original Noised Data")
plt.plot(xn2, func_exp(xn2, *popt2), 'r-', label="Fitted Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(xn3, yn3, 'ko', label="Original Noised Data")
plt.plot(xn3, func_exp(xn3, *popt3), 'r-', label="Fitted Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(xn4, yn4, 'ko', label="Original Noised Data")
plt.plot(xn4, func_exp(xn4, *popt4), 'r-', label="Fitted Curve")
plt.legend()
plt.show()


plt.figure()
plt.plot(np.array(processedTrace)+1)
plt.plot(xn0, yn0, 'ko')
plt.plot(xn0, func_exp(xn0, *popt0), 'r-')
plt.plot(xn1, yn1, 'ko')
plt.plot(xn1, func_exp(xn1, *popt1), 'r-')
plt.plot(xn2, yn2, 'ko')
plt.plot(xn2, func_exp(xn2, *popt2), 'r-')
plt.plot(xn3, yn3, 'ko')
plt.plot(xn3, func_exp(xn3, *popt3), 'r-')
plt.plot(xn4, yn4, 'ko')
plt.plot(xn4, func_exp(xn4, *popt4), 'r-')
plt.show()


exp0=np.array(func_exp(xn0, *popt0))
exp1=np.array(func_exp(xn1, *popt1))
exp2=np.array(func_exp(xn2, *popt2))
exp3=np.array(func_exp(xn3, *popt3))
exp4=np.array(func_exp(xn4, *popt4))

trans0 = np.append(np.array(processedTrace[16])+1,exp0)
trans1 = np.append(np.array(processedTrace[36])+1,exp1)
trans2 = np.append(np.array(processedTrace[56])+1,exp2)
trans3 = np.append(np.array(processedTrace[77])+1,exp3)
trans4 = np.append(np.array(processedTrace[97])+1,exp4)

fullTrace = np.append(trans0,trans1)
fullTrace = np.append(fullTrace,trans2)
fullTrace = np.append(fullTrace,trans3)
fullTrace = np.append(fullTrace,trans4)

ts=1/20
time = np.arange(ts*16, ts*(len(fullTrace)+16), ts)
time0 = np.arange(0, ts*len(processedTrace), ts)

peaks, _ = find_peaks(fullTrace, height=0)
results_half = peak_widths(fullTrace, peaks, rel_height=0.5)
results = np.array(results_half[0])*ts
print(results)

plt.figure()    
plt.plot(time[0:len(fullTrace)],fullTrace)
plt.plot(time0,np.array(processedTrace)+1)
 #   plt.plot(y1)
plt.plot((peaks+16)*ts, fullTrace[peaks], "x")
plt.hlines(*results_half, color="C2")




analysisPath=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\TemporalFWHM'


# time course figure
ts=1/20
time = np.arange(0, ts*len(processedTrace), ts)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(time,gf.norm(processedTrace_wf),linewidth=3.0,color='k',label='Widefield')
plt.plot(time,gf.norm(processedTrace_ref),linewidth=3.0,color='#049BEC',label='Refocused')
plt.plot(time,gf.norm(processedTrace_dec),linewidth=3.0,color='r',label='Deconvolved')
plt.plot([2.2,2.7],[0.9,0.9],linewidth=4.0,color='k')
fig.set_size_inches(10,4)
plt.legend(loc='upper left',frameon=False)

#axis formatting
plt.xlim([-0.15,2.8])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.savefig(analysisPath + r'\\temporalSpeadTime.png', format='png', dpi=600)
plt.savefig(analysisPath + r'\\temporalSpeadTime.eps', format='eps', dpi=800)
plt.close(fig)  




rf = [0.301752843, 0.20595586, 0.22656589, 0.277399523, 0.224987232]
dc = [0.144609936 , 0.227412298, 0.254392338, 0.095343157,0.241763366]


fig = plt.gcf()   
ax = plt.subplot(111)   
for point in range(len(rf)):
    plt.plot([1,2],[rf[point],dc[point]],marker='.',markersize='20',linewidth=3.0,color='k')
my_xticks = ['Refocused', 'Deconvolved']
x=np.array([1,2])
plt.xticks(x, my_xticks)
plt.ylabel('FWHM (s)', fontdict = font)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlim((0.75,2.25))
plt.tight_layout()
plt.savefig(analysisPath + r'\\temporalSpeadTime_all.png', format='png', dpi=600)
plt.savefig(analysisPath + r'\\temporalSpeadTime_all.eps', format='eps', dpi=800)
plt.close(fig)  
