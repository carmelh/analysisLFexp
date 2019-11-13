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


font = {'family': 'sans',
        'weight': 'normal',
        'size': 16,}

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
        df_REF1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_refocussed_{}.csv'.format(expDate),index_col = 'file name')        
        intra_REF.append(df_REF1)
        df_DEC1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_deconvolved_{}.csv'.format(expDate),index_col = 'file name')        
        intra_DEC.append(df_DEC1)        
        print('Intra')
        
    elif what == 'Bulk':
        print('Bulk')
        df_WF1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_WF_{}.csv'.format(expDate),index_col = 'file name')        
        bulk_WF.append(df_WF1)
        df_REF1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_refocussed_{}.csv'.format(expDate),index_col = 'file name')        
        bulk_REF.append(df_REF1)
        df_DEC1 = pd.read_csv(cwd + '\\' + what + '\\' + str(expDate) +r'\stats_deconvolved_{}.csv'.format(expDate),index_col = 'file name')        
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

##### FIGURES #####
#### Intra #########
SNR=[intra_WF['SNR'],intra_REF['SNR'],intra_DEC['SNR']]
fig=pf.boxPlot(SNR,'SNR')
pf.boxPlotMarkers(SNR)
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_SNR')
#pf.sigBars(intra_WF)

peakSig=[intra_WF['peakF'],intra_REF['peakF'],intra_DEC['peakF']]
pf.boxPlot(peakSig,'Peak Signal (%)')
pf.boxPlotMarkers(peakSig)
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_peak')
#pf.sigBars(intra_WF,'peakSignal')

noise=[intra_WF['dfNoise'],intra_REF['dfNoise'],intra_DEC['dfNoise']]
pf.boxPlot(noise,'Noise (%)')   
pf.boxPlotMarkers(noise)
pf.saveFigure(fig,analysisFolder + r'\Figures','intra_noise')
#pf.sigBars(intra_WF)
    

##### BULK #########    
SNR=[bulk_WF['SNR'],bulk_REF['SNR'],bulk_DEC['SNR']]
pf.boxPlot(SNR,'SNR')
pf.boxPlotMarkers(SNR)
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_SNR')

peakSig=[bulk_WF['peakF'],bulk_REF['peakF'],bulk_DEC['peakF']]
pf.boxPlot(peakSig,'Peak Signal (%)')
pf.boxPlotMarkers(peakSig)
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_peak')

noise=[bulk_WF['dfNoise'],bulk_REF['dfNoise'],bulk_DEC['dfNoise']]
pf.boxPlot(noise,'Noise (%)')
pf.boxPlotMarkers(noise)
pf.saveFigure(fig,analysisFolder + r'\Figures','bulk_noise')




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
