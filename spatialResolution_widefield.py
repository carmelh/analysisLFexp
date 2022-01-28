# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:00:16 2020

@author: chowe7
"""

import numpy as np
import tifffile
import pyqtgraph as pg
import sys
import pandas as pd
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
import general_functions as gf
import matplotlib.pyplot as plt
import plotting_functions as pf
from scipy.interpolate import UnivariateSpline

font = {'family': 'sans',
        'weight': 'normal',
        'size': 16,}

plt.rc('font',**font)    

date=190605
analysisFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Analysis'
path = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190730\slice2\cell1\z_stack\WF3_1x1_50ms_100pA_2'
fileName = '\\WF3_1x1_50ms_100pA_2_MMStack_Pos0.ome.tif'
figureFolder = path + '\\figures'
stack = tifffile.imread(path + fileName)


pg.image(stack)
z = 44
plt.imshow(stack[z,...])

pixelSize = 0.26
xLat=np.linspace(0, stack.shape[1]*pixelSize, stack.shape[2])  #x in um
xAx=np.linspace(0, stack.shape[0], stack.shape[0])  #x in um

xLoc=1260
yLoc=888

xy=stack[z,xLoc,:]
yx=stack[z,:,yLoc]

fig = plt.gcf()    
ax = fig.add_subplot(111)
plt.plot(xLat,xy,linewidth=2.5,color='k')
    
ax.set_xlabel('x (um)')    
ax.set_xlim((180,300))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\xy.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xy.eps', format='eps', dpi=600, bbox_inches='tight')



fig = plt.gcf()    
ax = fig.add_subplot(111)
plt.plot(xLat,yx,linewidth=2.5, color='k')
ax.set_xlabel('x (um)')    
ax.set_xlim((300,360))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
    
plt.savefig(figureFolder + r'\\yx.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yx.eps', format='eps', dpi=600, bbox_inches='tight')

# FOV image with scale bar
xS=250
lengthSB = 25
y = 330

upSamp = round((20*5)/pixelSize)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(stack[z,round(xLoc-(upSamp/2)):round(xLoc+(upSamp/2)),round(yLoc-(upSamp/2)):round(yLoc+(upSamp/2))])
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
plt.savefig(figureFolder + r'\\FOV_25umSB.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_25umSB.eps', format='eps', dpi=800, bbox_inches='tight')


#########################
#     Lateral FWHM      #
#########################

spline = UnivariateSpline(xLat, xy-np.max(xy)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xy=r2-r1


spline = UnivariateSpline(xLat, yx-np.max(yx)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yx=r2-r1

fields=[date,FWHM_xy,FWHM_yx,xLoc,yLoc,figureFolder]
gf.appendCSV(analysisFolder ,r'\lateralResolution_widefield_FWHM',fields)



################################
#      Axial Resolution        #
################################

# xz and yz are 2d images through z. 
xz=stack[:,xLoc,:]
yz=stack[:,:,yLoc]


#########################
#      Axial FWHM       #
#########################

y=xz[:,yLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xz=r2-r1

y=yz[:,xLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yz=r2-r1

fields=[date,FWHM_xz,FWHM_yz,xLoc,yLoc,figureFolder]
gf.appendCSV(analysisFolder ,r'\axialResolution_widefield_FWHM',fields)



fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(xz[:,round(yLoc-(upSamp/2)):round(yLoc+(upSamp/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_80um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(yz[:,round(xLoc-(upSamp/2)):round(xLoc+(upSamp/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_80um.eps', format='eps', dpi=800, bbox_inches='tight')






