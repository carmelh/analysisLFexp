# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:00:03 2019

@author: chowe7
"""

# to do:
#   pandas read in of data
#   stats on the stats from that

import numpy as np
import sys
import pandas as pd
#sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
import imagingAnalysis as ia
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import matplotlib.pyplot as plt

font = {'family': 'sans',
        'weight': 'normal',
        'size': 16,}

plt.rc('font',**font)


def getPowerDensity(power_cal,df):
    for idx in range(len(df)):
        currentFile = df.index.values[idx]
        print('file {}'.format(currentFile))
        try:
            LEDcurrent = df.at[currentFile, 'LED (A)']
            pd_df = power_cal[power_cal['current']==LEDcurrent]      
            pd = pd_df.at[np.int(pd_df.index.values),'Power density (mW/mm2)']
            df.at[currentFile, 'LED (PD)'] = pd
        
        except:
            df.at[currentFile, 'LED (PD)'] = 'nan'
            print('LED Power not found')
    return 


date = '190724'
cwd = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190724'

df_WF = pd.read_csv(cwd+r'\stats_WF_{}.csv'.format(date),index_col = 'file name')
df_REF = pd.read_csv(cwd+r'\stats_refocussed_{}.csv'.format(date),index_col = 'file name')
df_DEC = pd.read_csv(cwd+r'\stats_deconvolved_{}.csv'.format(date),index_col = 'file name')


data_summary = pd.ExcelFile(cwd+r'\Setup\power_cal_{}.xlsx'.format(date))
power_cal = data_summary.parse('Sheet1')    


getPowerDensity(power_cal,df_REF)
getPowerDensity(power_cal,df_WF)
getPowerDensity(power_cal,df_DEC)

refPD = df_REF.groupby('LED (PD)').mean()
wfPD = df_WF.groupby('LED (PD)').mean()
decPD = df_DEC.groupby('LED (PD)').mean()


date2 = '190730'
cwd2 = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190730'

df_WF2 = pd.read_csv(cwd2+r'\stats_WF_{}.csv'.format(date2),index_col = 'file name')
df_REF2 = pd.read_csv(cwd2+r'\stats_refocussed_{}.csv'.format(date2),index_col = 'file name')
df_DEC2 = pd.read_csv(cwd2+r'\stats_deconvolved_{}.csv'.format(date2),index_col = 'file name')


data_summary2 = pd.ExcelFile(cwd2+r'\Setup\power_cal_{}.xlsx'.format(date2))
power_cal2 = data_summary2.parse('Sheet1')    

getPowerDensity(power_cal2,df_REF2)
getPowerDensity(power_cal2,df_WF2)
getPowerDensity(power_cal2,df_DEC2)

refPD2 = df_REF2.groupby('LED (PD)').mean()
wfPD2 = df_WF2.groupby('LED (PD)').mean()
decPD2 = df_DEC2.groupby('LED (PD)').mean()



## PLOTS ##

## SNR ##
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(wfPD.index,wfPD['SNR'],'o-',linewidth=4.0,color='k',markersize=12, label='Widefield')
ax.plot(refPD.index,refPD['SNR'],'o-',linewidth=4.0,color='r',markersize=12, label='Refocussed')
ax.plot(decPD.index,decPD['SNR'],'o-',linewidth=4.0,color='#104FBC',markersize=12,label='Deconvolved')
        
#ax.plot(wfPD2.index,wfPD2['SNR'],'o-',linewidth=4.0,color='k',markersize=12, label='Widefield')
ax.plot(refPD2.index,refPD2['SNR'],'o-',linewidth=4.0,color='r',markersize=12, label='Refocussed')
ax.plot(decPD2.index,decPD2['SNR'],'o-',linewidth=4.0,color='#104FBC',markersize=12,label='Deconvolved')
        
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel('Power Density (mW/mm2)', fontdict = font)
plt.ylabel('SNR', fontdict = font)
ax.legend(frameon=False)
plt.tight_layout()

## % noise ##
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(wfPD.index,wfPD['df noise'],'o-',linewidth=4.0,color='k',markersize=12,label='Widefield')
ax.plot(refPD.index,refPD['dF noise'],'o-',linewidth=4.0,color='r',markersize=12,label='Refocussed')
ax.plot(decPD.index,decPD['dF noise'],'o-',linewidth=4.0,color='#104FBC',markersize=12,label='Deconvolved')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel('Power Density (mW/mm2)', fontdict = font)
plt.ylabel('Noise (%)', fontdict = font)
ax.legend(frameon=False)
plt.tight_layout()

## peak dF/F % ##
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(wfPD.index,wfPD['peak dF/F'],'o-',linewidth=4.0,color='k',markersize=12,label='Widefield')
ax.plot(refPD.index,refPD['peak dFF'],'o-',linewidth=4.0,color='r',markersize=12,label='Refocussed')
ax.plot(decPD.index,decPD['peak dFF'],'o-',linewidth=4.0,color='#104FBC',markersize=12,label='Deconvolved')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xlabel('Power Density (mW/mm2)', fontdict = font)
plt.ylabel('Peak dF/F (%)', fontdict = font)
ax.legend(frameon=False)
plt.tight_layout()