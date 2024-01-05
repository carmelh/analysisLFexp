# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 17:58:57 2019

@author: chowe7
"""

import numpy as np
import sys
import pandas as pd
#sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import plotting_functions as pf
import matplotlib.pyplot as plt
from datetime import date
from scipy.stats import wilcoxon


font = {'family': 'sans',
        'weight': 'normal',
        'size': 25,}

plt.rc('font',**font)


### SETUP ###
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1' 
analysisFolder = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis'
today = date.today()
today = today.strftime("%y%m%d")

df = pd.read_excel(cwd+r'\data_summary.xlsx',index_col = 'Date')

intra_WF = []
intra_REF = []
intra_DEC =[]
bulk_WF = []
bulk_REF = []
bulk_DEC =[]


########################### 
## Import all experiment day results ##
for exp in range(len(df)):
    expDate = df.index.values[exp]
    print(expDate)
    what = '{}'.format(df.at[expDate, 'What'])
    
    if what == 'Intra':
        df_WF1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_WF_{}.csv'.format(expDate),index_col = 'file name')        
        intra_WF.append(df_WF1)
        df_REF1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_refocused_wProcesses_{}.csv'.format(expDate),index_col = 'file name')        
        intra_REF.append(df_REF1)
        df_DEC1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_deconvolved_660_wProcesses_{}.csv'.format(expDate),index_col = 'file name')        
        intra_DEC.append(df_DEC1)        
        print('Intra')
        
    elif what == 'Bulk':
        print('Bulk')
        df_WF1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_WF_{}.csv'.format(expDate),index_col = 'file name')        
        bulk_WF.append(df_WF1)
        df_REF1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_refocused_{}.csv'.format(expDate),index_col = 'file name')        
        bulk_REF.append(df_REF1)
        df_DEC1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +
                              r'\stats_deconvolved_{}.csv'.format(expDate),index_col = 'file name')        
        bulk_DEC.append(df_DEC1)    

intra_WF = pd.concat(intra_WF,sort=True)    
intra_REF = pd.concat(intra_REF,sort=True)     
intra_DEC = pd.concat(intra_DEC,sort=True)   
bulk_WF = pd.concat(bulk_WF,sort=True)    
bulk_REF = pd.concat(bulk_REF,sort=True)    
bulk_DEC = pd.concat(bulk_DEC,sort=True)   


##################################################### 
# save stats on stats ###########
##### INTRAA #########
intra_WF_stats = pd.concat([intra_WF.median(), intra_WF.quantile(0.9),intra_WF.quantile(0.1),intra_WF.count()],
                axis=1,sort=False).reindex(intra_WF.median().index)
intra_WF_stats.columns=['Median', '90th','10th','Total']
intra_REF_stats = pd.concat([intra_REF.median(), intra_REF.quantile(0.9),intra_REF.quantile(0.1),intra_REF.count()],
                axis=1,sort=False).reindex(intra_REF.median().index)
intra_REF_stats.columns=['Median', '90th','10th','Total']
intra_DEC_stats = pd.concat([intra_DEC.median(), intra_DEC.quantile(0.9),intra_DEC.quantile(0.1),intra_DEC.count()],
                axis=1,sort=False).reindex(intra_DEC.median().index)
intra_DEC_stats.columns=['Median', '90th','10th','Total']

with pd.ExcelWriter(analysisFolder + r'\statsOnStats_intra_{}.xlsx'.format(today)) as writer: 
    intra_WF_stats.to_excel(writer, sheet_name='WF')
    intra_REF_stats.to_excel(writer, sheet_name='Refocused')
    intra_DEC_stats.to_excel(writer, sheet_name='Deconvolved')
 
    
w_SNR, p_SNR = wilcoxon(intra_REF['SNR'],intra_DEC['SNR'])
print('w = ', w_SNR, 'p = ', p_SNR)
w_Peak, p_Peak = wilcoxon(intra_REF['peakF'],intra_DEC['peakF'])
print('w = ', w_Peak, 'p = ', p_Peak)
w_Noise, p_Noise = wilcoxon(intra_REF['dfNoise'],intra_DEC['dfNoise'])
print('w = ', w_Noise, 'p = ', p_Noise)


##### BULK #########
bulk_WF_stats = pd.concat([bulk_WF.median(), bulk_WF.quantile(0.9),bulk_WF.quantile(0.1),bulk_WF.count()],
                axis=1,sort=False).reindex(bulk_WF.median().index)
bulk_WF_stats.columns=['Median', '90th','10th','Total']
bulk_REF_stats = pd.concat([bulk_REF.median(), bulk_REF.quantile(0.9),bulk_REF.quantile(0.1),bulk_REF.count()],
                axis=1,sort=False).reindex(bulk_REF.median().index)
bulk_REF_stats.columns=['Median', '90th','10th','Total']
bulk_DEC_stats = pd.concat([bulk_DEC.median(), bulk_DEC.quantile(0.9),bulk_DEC.quantile(0.1),bulk_DEC.count()],
                axis=1,sort=False).reindex(bulk_DEC.median().index)
bulk_DEC_stats.columns=['Median', '90th','10th','Total']

with pd.ExcelWriter(analysisFolder + r'\statsOnStats_bulk_{}.xlsx'.format(today)) as writer: 
    bulk_WF_stats.to_excel(writer, sheet_name='WF')
    bulk_REF_stats.to_excel(writer, sheet_name='Refocused')
    bulk_DEC_stats.to_excel(writer, sheet_name='Deconvolved')
    
    
########################### 


w_SNR, p_SNR = wilcoxon(bulk_REF['SNR'],bulk_DEC['SNR'])
print('w = ', w_SNR, 'p = ', p_SNR)
w_Peak, p_Peak = wilcoxon(bulk_REF['peakF'],bulk_DEC['peakF'])
print('w = ', w_Peak, 'p = ', p_Peak)
w_Noise, p_Noise = wilcoxon(bulk_REF['dfNoise'],bulk_DEC['dfNoise'])
print('w = ', w_Noise, 'p = ', p_Noise)


##### FIGURES #####
#### Intra #########
SNR=[intra_WF['SNR'],intra_REF['SNR'],intra_DEC['SNR']]
fig=pf.boxPlot(SNR,'SNR')
plt.xticks(rotation=-30)
pf.boxPlotMarkers(SNR)
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_SNR_incProcesses')
#pf.sigBars(intra_WF)

peakSig=[intra_WF['peakF'],intra_REF['peakF'],intra_DEC['peakF']]
fig=pf.boxPlot(peakSig,'Peak Signal (%)')
plt.xticks(rotation=-30)
pf.boxPlotMarkers(peakSig)
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_peak_incProcesses')
#pf.sigBars(intra_WF,'peakSignal')

noise=[intra_WF['dfNoise'],intra_REF['dfNoise'],intra_DEC['dfNoise']]
fig=pf.boxPlot(noise,'Noise (%)')  
plt.xticks(rotation=-30) 
pf.boxPlotMarkers(noise)
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_noise_incProcesses')
#pf.sigBars(intra_WF)
 

# TESTT! Paired figire + Box
import pylab as P

noise=[intra_WF['dfNoise'],intra_REF['dfNoise'],intra_DEC['dfNoise']]
boxprops = dict(linewidth=2)
medianprops = dict(linewidth=2.5,color='DarkCyan')
flierprops = dict(markersize=10,linestyle='none')

fig = plt.figure()
ax = fig.add_subplot(111)
for trial in range(len(intra_REF)):
    plt.plot([2,3],[intra_REF['dfNoise'][trial],intra_DEC['dfNoise'][trial]],marker='.',markersize='14',color='darkgrey',linewidth=3)
y = intra_WF['dfNoise']
x = np.random.normal(1, 0.04, size=len(y))
P.plot(x, y, '.', color='darkgrey',markersize=14)
bp=plt.boxplot(noise,boxprops=boxprops,medianprops=medianprops,flierprops=flierprops)
my_xticks = ['WF', 'Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('{}'.format('Noise (%)'), fontdict = font)
   # plt.xlim((0.5,3.1))
plt.tight_layout()
for whisker in bp['whiskers']:
    whisker.set(linewidth=3)
for cap in bp['caps']:
    cap.set(linewidth=3)
plt.xticks(rotation=-30) 
fig.set_size_inches(7,6)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_noise_boxPaired_wProcesses')


# Paired figire tutto 
fig = plt.figure()
ax3 = fig.add_subplot(133)
plt.plot(np.ones(len(intra_WF['SNR'])),intra_WF['SNR'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(intra_REF)):
    plt.plot([2,3],[intra_REF['SNR'][trial],intra_DEC['SNR'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_linewidth(2)
ax3.spines['left'].set_linewidth(2)
plt.ylabel('SNR', fontdict = font)
plt.xlim((0.65,3.35))

ax1 = fig.add_subplot(131)
plt.plot(np.ones(len(intra_WF['peakF'])),intra_WF['peakF'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(intra_REF)):
    plt.plot([2,3],[intra_REF['peakF'][trial],intra_DEC['peakF'][trial]],marker='.',markersize='24',color='k',linewidth=3)
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
plt.ylabel('Peak Signal (%)', fontdict = font)
plt.xlim((0.65,3.35))

ax2 = fig.add_subplot(132)
plt.plot(np.ones(len(intra_WF['dfNoise'])),intra_WF['dfNoise'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(intra_REF)):
    plt.plot([2,3],[intra_REF['dfNoise'][trial],intra_DEC['dfNoise'][trial]],marker='.',markersize='24',color='k',linewidth=3)
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.ylabel('Noise (%)', fontdict = font)
plt.xlim((0.65,3.35))

fig.set_size_inches(20,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','tutto_intra')

#bulk tutto
fig = plt.figure()
ax3 = fig.add_subplot(133)
plt.plot(np.ones(len(bulk_WF['SNR'])),bulk_WF['SNR'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(bulk_REF)):
    plt.plot([2,3],[bulk_REF['SNR'][trial],bulk_DEC['SNR'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_linewidth(2)
ax3.spines['left'].set_linewidth(2)
plt.ylabel('SNR', fontdict = font)
plt.xlim((0.65,3.35))

ax1 = fig.add_subplot(131)
plt.plot(np.ones(len(bulk_WF['peakF'])),bulk_WF['peakF'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(bulk_REF)):
    plt.plot([2,3],[bulk_REF['peakF'][trial],bulk_DEC['peakF'][trial]],marker='.',markersize='24',color='k',linewidth=3)
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['left'].set_linewidth(2)
plt.ylabel('Peak Signal (%)', fontdict = font)
plt.xlim((0.65,3.35))

ax2 = fig.add_subplot(132)
plt.plot(np.ones(len(bulk_WF['dfNoise'])),bulk_WF['dfNoise'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(bulk_REF)):
    plt.plot([2,3],[bulk_REF['dfNoise'][trial],bulk_DEC['dfNoise'][trial]],marker='.',markersize='24',color='k',linewidth=3)
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['left'].set_linewidth(2)
plt.ylabel('Noise (%)', fontdict = font)
plt.xlim((0.65,3.35))

fig.set_size_inches(20,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','tutto_bulk')



# Paired figire
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.ones(len(intra_WF['SNR'])),intra_WF['SNR'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(intra_REF)):
    plt.plot([2,3],[intra_REF['SNR'][trial],intra_DEC['SNR'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('SNR', fontdict = font)
plt.xlim((0.65,3.35))
fig.set_size_inches(7,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_SNR_paired_wProcesses')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.ones(len(intra_WF['peakF'])),intra_WF['peakF'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(intra_REF)):
    plt.plot([2,3],[intra_REF['peakF'][trial],intra_DEC['peakF'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('Peak Signal (%)', fontdict = font)
plt.xlim((0.65,3.35))
fig.set_size_inches(7,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_peak_paired_wProcesses')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.ones(len(intra_WF['dfNoise'])),intra_WF['dfNoise'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(intra_REF)):
    plt.plot([2,3],[intra_REF['dfNoise'][trial]/100,intra_DEC['dfNoise'][trial]/100],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('Noise (%)', fontdict = font)
plt.xlim((0.65,3.35))
fig.set_size_inches(7,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_noise_paired_wProcesses')


##### BULK #########    
SNR=[bulk_WF['SNR'],bulk_REF['SNR'],bulk_DEC['SNR']]
fig=pf.boxPlot(SNR,'SNR')
pf.boxPlotMarkers(SNR)
plt.xticks(rotation=-30) 
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_SNR')

peakSig=[bulk_WF['peakF'],bulk_REF['peakF'],bulk_DEC['peakF']]
fig=pf.boxPlot(peakSig,'Peak Signal (%)')
pf.boxPlotMarkers(peakSig)
plt.xticks(rotation=-30) 
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_peak')

noise=[bulk_WF['dfNoise'],bulk_REF['dfNoise'],bulk_DEC['dfNoise']]
fig=pf.boxPlot(noise,'Noise (%)')
pf.boxPlotMarkers(noise)
plt.xticks(rotation=-30) 
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_noise')


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.ones(len(bulk_WF['SNR'])),bulk_WF['SNR'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(bulk_REF)):
    plt.plot([2,3],[bulk_REF['SNR'][trial],bulk_DEC['SNR'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('SNR', fontdict = font)
plt.xlim((0.65,3.35))
fig.set_size_inches(7,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_SNR_paired')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.ones(len(bulk_WF['peakF'])),bulk_WF['peakF'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(bulk_REF)):
    plt.plot([2,3],[bulk_REF['peakF'][trial],bulk_DEC['peakF'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('Peak Signal (%)', fontdict = font)
plt.xlim((0.65,3.35))
fig.set_size_inches(7,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_peak_paired')

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.ones(len(bulk_WF['dfNoise'])),bulk_WF['dfNoise'],linestyle='None',color='k',marker='.',markersize='24')
for trial in range(len(bulk_REF)):
    plt.plot([2,3],[bulk_REF['dfNoise'][trial],bulk_DEC['dfNoise'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('Noise (%)', fontdict = font)
plt.xlim((0.65,3.35))
fig.set_size_inches(7,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_noise_paired')

#
#import matplotlib.pyplot as plt
#import seaborn as sns
#from statannot import add_stat_annotation
#
#sns.set(style="whitegrid")
#df = sns.load_dataset("tips")
#
#x = "day"
#y = "total_bill"
#order = ['Sun', 'Thur', 'Fri', 'Sat']
#ax = sns.boxplot(data=df, x=x, y=y, order=order)
#add_stat_annotation(ax, data=df, x=x, y=y, order=order,
#                    box_pairs=[("Thur", "Fri"), ("Thur", "Sat"), ("Fri", "Sun")],
#                    test='Mann-Whitney', text_format='star', loc='outside', verbose=2)


fig = plt.figure()
ax = fig.add_subplot(111)
for trial in range(len(intra_REF)):
    plt.plot([1,2,3],[intra_WF['SNR'][trial],intra_REF['SNR'][trial],intra_DEC['SNR'][trial]],marker='.',markersize='24',color='k',linewidth=3)
my_xticks = ['WF','Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.ylabel('SNR', fontdict = font)
plt.xlim((0.85,3.15))
fig.set_size_inches(7,5)
plt.tight_layout()
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_SNR_paired')

