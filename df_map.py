# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 15:03:00 2020

@author: chowe7
"""

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

keyword = 'refocused'

if keyword == 'refocused':
    file= refoc_mean_stack
    darkTrialAverage =90
elif keyword == 'deconvolved':
    file= decon_mean_stack
    darkTrialAverage=0



back=np.mean(file[:13,...],0)
df=100*(file-back[None,...]) / (back[None,...] - darkTrialAverage)




df=100*(file-back) / (back - darkTrialAverage)







#result = np.zeros((80,101,101))
#df = np.zeros((200,80,101,101))
#
#for z in range(80):
#    baselineFluorescence = np.mean(file[0:13,z,...],0)
#    print(z)
## now calc (f-f0)/(f0-fdark)     
#    for x in range(101):
#        for y in range(101):
##        processedTrace.append((element-baselineFluorescence)/(baselineFluorescence-darkTrialAverage))
#            element = file[:,z,x,y]
#            df[:,z,x,y]=(element-baselineFluorescence)/(baselineFluorescence-darkTrialAverage)
#            result[z,x,y]=np.std((element-baselineFluorescence)/(baselineFluorescence-darkTrialAverage))
#        
        
    #for col,ii in zip(color_idx, range(len(trialData_soma))):
#    lines=plt.plot(trialData_soma[ii],linewidth=2.5, color=plt.cm.plasma(col))
    
#for static fov images
lengthSB = 25
pixelSize = 5
xS = 25
y = 31

fig = plt.gcf()      
ax = plt.subplot(111)  
im = plt.imshow(result[40,45:79,30:64]*100)
plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
forceAspect(ax,aspect=1)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\dfImage_{}_SB-25um.png'.format(keyword), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\dfImage_{}_SB-25um.eps'.format(keyword), format='eps', dpi=800, bbox_inches='tight')



xS = 49
y = 55
figureFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4\figures\3D\z_stack'

vmin=np.min(df[17,:,30:90,20:80])
vmax=np.max(df[17,:,30:90,20:80])

for z in range(81):
    fig = plt.gcf()      
    ax = plt.subplot(111)  
    im = plt.imshow(df[17,z,30:90,20:80],vmin=vmin,vmax=vmax,cmap='gray')
    plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    forceAspect(ax,aspect=1)
    plt.tight_layout()  
    number_str = str(z)
    zero_filled_number = number_str.zfill(3)
    print(zero_filled_number)
    plt.savefig(figureFolder + r'\\dfImageNormGray_z-{}_SB-25um.png'.format(zero_filled_number), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)   

    


