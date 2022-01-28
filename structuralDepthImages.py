# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 12:22:29 2022

@author: chowe7
"""

import numpy as np
import sys
import pandas as pd
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import matplotlib.pyplot as plt
import plotting_functions as pf
from scipy.interpolate import UnivariateSpline
import pyqtgraph as pg

font = {'family': 'sans',
        'weight': 'normal',
        'size': 20,}

plt.rc('font',**font) 

path=r'Y:\projects\thefarm2\ephemeral\figures\depth_from190605_2'

dms_3=decon_mean_stack[1]

xLoc=47
yLoc=64


lengthSB=25
pixelSize=5
xS=32
y=38


df_dec=np.zeros((82,122,101,101))
var3d_dec=np.zeros((len(df_dec),len(df_dec[0][0]),len(df_dec[0][1])))
for z in range(len(dms_3)):
    file=dms_3[z]
    back=np.mean(file[:13,...],0) 
    df_dec[z]=100*(file-back[None,...]) / (back[None,...] - 0)
    var3d_dec[z]= np.var(file,axis=0)



for z in range(len(dms_3)):
    print(z)
    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.imshow(df_dec[z,17,yLoc-21:yLoc+21,xLoc-21:xLoc+21],cmap='gray')
    plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    pf.forceAspect(ax,aspect=1)
    plt.tight_layout()      
    plt.savefig(path + r'\\func\\funcFOV_decon3it_gray_z-{}_25umSB.png'.format(z), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)

    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.imshow(dms_3[z,17,yLoc-21:yLoc+21,xLoc-21:xLoc+21],cmap='gray')
    plt.plot([xS,xS+(lengthSB/pixelSize)],[y,y],linewidth=6.0,color='w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    pf.forceAspect(ax,aspect=1)
    plt.tight_layout()      
    plt.savefig(path + r'\\struct\\structFOV_decon3it_gray_z-{}_25umSB.png'.format(z), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)
    
#     WF
import tifffile

stack = tifffile.imread(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ZSTCK\NOMLA_NOPIP_1x1_f_2-8_660nm_200mA_2\NOMLA_NOPIP_1x1_f_2-8_660nm_200mA_2_MMStack_Pos0.ome.tif')

infocus=49

xLocUp=round((2048/101)*88)
yLocUp=round((2048/101)*yLoc)
diff=round((2048/101)*21)

xS=652
y=762

sameRange=stack[infocus-41:infocus+41]

for z in range(len(sameRange)):
    print(z)
    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.imshow(sameRange[z,1259-diff:1259+diff,889-diff:889+diff],cmap='gray')
    plt.plot([xS,xS+(25/0.26)],[y,y],linewidth=6.0,color='w')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    pf.forceAspect(ax,aspect=1)
    plt.tight_layout()      
    plt.savefig(path + r'\\wf\\staticFOV_wf_gray_z-{}_25umSB.png'.format(z), format='png', dpi=600, bbox_inches='tight')
    plt.close(fig)


