# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 17:29:43 2020

@author: chowe7
"""

import numpy as np
import pyqtgraph as pg
import sys
import pandas as pd
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import matplotlib.pyplot as plt
import plotting_functions as pf

font = {'family': 'sans',
        'weight': 'normal',
        'size': 20,}

plt.rc('font',**font)    

analysisFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\spatialRes'
figureFolder = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis\spatialRes\Figures'


##############################
#      Summary Figures       #
##############################

keyword = 'lateral' # axial or lateral
df = pd.read_csv(analysisFolder + '\\{}Resolution_deconvolved_FWHM.csv'.format(keyword))
df=df.sort_values(by=['num it'])
df_ref = pd.read_csv(analysisFolder + '\\{}Resolution_refocused_FWHM.csv'.format(keyword))
df_wf = pd.read_csv(analysisFolder + '\\{}Resolution_widefield_FWHM.csv'.format(keyword))

# 550 nm
#zExp_190605 = [36.90748133,17.65256365,10.31410597]
#zExp_190724 = [17.89652043,8.72784200434917,5.55935344342998]
#zExp_190730 = [24.28052227,11.7886567331751,7.351862839]
#
#
#xyExp_190605 = [21.92377427,16.13085391,7.342852238]
#xyExp_190724 = [6.340962823,9.98803646969958,6.61511070421352]
#xyExp_190730 = [10.33627753,13.5215534313085,7.824602214]
#
#
#yxExp_190605 = [23.66324225,15.56676201,9.003281388]
#yxExp_190724 = [8.02144382,8.1609340849707,6.07529136403994]
#yxExp_190730 = [13.09783772,10.7265328213096,7.011397032]

#660 wf, ref, decon
zExp_190605 = [36.90748133,17.89338287,12.9164624256418]
zExp_190724 = [17.89652043,9.472001856,5.46946373]
zExp_190730 = [24.28052227,15.10441696,10.3766878]


xyExp_190605 = [21.92377427,15.3494019221694,11.4847233917301]
xyExp_190724 = [6.340962823,9.472001856,6.566136246]
xyExp_190730 = [10.33627753,13.52089592,10.14538913]


yxExp_190605 = [23.66324225,13.86161501447351,9.93131382]
yxExp_190724 = [8.02144382,7.972254281,6.15576694]
yxExp_190730 = [13.09783772,9.070550852,6.528911556]


zExp21_190605 = [17.89338287,8.99047387796438]
zExp21_190724 = [9.472001856,3.654388061]
zExp21_190730 = [15.10441696,5.997520894]

# ref to 21 it
yExp21_190605 = [13.86161501447351,7.610966943]
yExp21_190724 = [7.972254281,5.970137118]
yExp21_190730 = [9.070550852,5.997520894]


ref_z=zExp_190605[1],zExp_190724[1],zExp_190730[1]
dec_z=zExp_190605[2],zExp_190724[2],zExp_190730[2]
print('Ref',np.median(ref_z),np.percentile(ref_z,10),np.percentile(ref_z,90))
print('Dec',np.median(dec_z),np.percentile(dec_z,10),np.percentile(dec_z,90))

ref_x=xyExp_190605[1],xyExp_190724[1],xyExp_190730[1]
dec_x=xyExp_190605[2],xyExp_190724[2],xyExp_190730[2]
print('Ref',np.median(ref_x),np.percentile(ref_x,10),np.percentile(ref_x,90))
print('Dec',np.median(dec_x),np.percentile(dec_x,10),np.percentile(dec_x,90))

ref_y=yxExp_190605[1],yxExp_190724[1],yxExp_190730[1]
dec_y=yxExp_190605[2],yxExp_190724[2],yxExp_190730[2]
print('Ref',np.median(ref_y),np.percentile(ref_y,10),np.percentile(ref_y,90))
print('Dec',np.median(dec_y),np.percentile(dec_y,10),np.percentile(dec_y,90))

fig = plt.gcf()      
ax = plt.subplot(111)
plt.plot(np.arange(2,4,1),zExp21_190605,color='r',linewidth='3',linestyle='--',marker='.',markersize='14')
plt.plot(np.arange(2,4,1),zExp21_190724,color='r',linewidth='3',linestyle='--',marker='.',markersize='14')
plt.plot(np.arange(2,4,1),zExp21_190730,color='r',linewidth='3',linestyle='--',marker='.',markersize='14')

plt.plot(np.arange(1,4,1),zExp_190605,color='k',marker='.',markersize='14',linewidth='3')
plt.plot(np.arange(1,4,1),zExp_190724,color='k',marker='.',markersize='14',linewidth='3')
plt.plot(np.arange(1,4,1),zExp_190730,color='k',marker='.',markersize='14',linewidth='3')

my_xticks = ['Widefield', 'Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
#plt.xticks(rotation=-30) 
plt.ylabel('$z_{FWHM}$ (\u03bcm)', fontdict = font)
ax.set_xlim((0.8,3.2))
fig.set_size_inches(6,5.5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\allAxialResolution_summaryInc21.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\allAxialResolution_summaryInc21.eps'.format(keyword), format='eps', dpi=800, bbox_inches='tight')


# y with 21
fig = plt.gcf()      
ax = plt.subplot(111)
plt.plot(np.arange(2,4,1),yExp21_190605,color='r',linewidth='3',linestyle='--',marker='.',markersize='14')
plt.plot(np.arange(2,4,1),yExp21_190724,color='r',linewidth='3',linestyle='--',marker='.',markersize='14')
plt.plot(np.arange(2,4,1),yExp21_190730,color='r',linewidth='3',linestyle='--',marker='.',markersize='14')
plt.plot(np.arange(1,4,1),yxExp_190605,color='k',marker='.',markersize='14',linewidth='3',label='yx')
plt.plot(np.arange(1,4,1),yxExp_190724,color='k',marker='.',markersize='14',linewidth='3')
plt.plot(np.arange(1,4,1),yxExp_190730,color='k',marker='.',markersize='14',linewidth='3')

my_xticks = ['Widefield', 'Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
#plt.legend(frameon=False)
#axis formatting
plt.yticks(np.arange(5, 26, step=5)) 
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
#plt.xticks(rotation=-30) 
plt.ylabel('$y_{FWHM}$ (\u03bcm)', fontdict = font)
ax.set_xlim((0.8,3.2))
fig.set_size_inches(6,5.5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yLateralResolution_summaryInc21.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yLateralResolution_summaryInc21.eps'.format(keyword), format='eps', dpi=800, bbox_inches='tight')



# x and y
fig = plt.gcf()      
ax = plt.subplot(111)
plt.plot(np.arange(1,4,1),xyExp_190605,color='k',marker='.',markersize='14',linewidth='3',label='xy')
plt.plot(np.arange(1,4,1),xyExp_190724,color='k',marker='.',markersize='14',linewidth='3')
plt.plot(np.arange(1,4,1),xyExp_190730,color='k',marker='.',markersize='14',linewidth='3')
plt.plot(np.arange(1,4,1),yxExp_190605,color='darkgrey',marker='.',markersize='14',linewidth='3',label='yx')
plt.plot(np.arange(1,4,1),yxExp_190724,color='darkgrey',marker='.',markersize='14',linewidth='3')
plt.plot(np.arange(1,4,1),yxExp_190730,color='darkgrey',marker='.',markersize='14',linewidth='3')

my_xticks = ['Widefield', 'Refoc.', 'Decon.']
x=np.array([1,2,3])
plt.xticks(x, my_xticks)
plt.legend(frameon=False)
#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
plt.xticks(rotation=-30) 
plt.ylabel('$xy_{res}$ (\u03bcm)', fontdict = font)
ax.set_xlim((0.8,3.2))
fig.set_size_inches(6,5.5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\allLateralResolution_summary.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\allLateralResolution_summary.eps'.format(keyword), format='eps', dpi=800, bbox_inches='tight')
