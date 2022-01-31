# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 09:27:50 2022

@author: chowe7
"""

import numpy as np
import tifffile
import sys
sys.path.insert(1, r'H:\Python_Scripts\analysisLFexp')
sys.path.insert(1, r'H:\Python_Scripts\carmel_functions')
sys.path.insert(1, r'\\icnas4.cc.ic.ac.uk\chowe7\GitHub\lightfield_HPC_processing')
import deconvolve as de
import general_functions as gf
import plotting_functions as pf
from scipy.interpolate import UnivariateSpline
import pyqtgraph as pg
import matplotlib.pyplot as plt

font = {'family': 'sans',
        'weight': 'normal',
        'size': 20,}

plt.rc('font',**font)   

figureFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Phantoms\NonScattering\10umBeads_best\lf_refl660nm_stack_1\z18\figures'

pixelSize=5

result_decon=decon_mean_stack
xLat=np.linspace(0, result_decon.shape[3]*pixelSize, result_decon.shape[3])  #x in um
depths_dec = np.arange(-40,41,1)
depths_ref = np.arange(-40,41,1)

#z18
xLoc=51
yLoc=67
# z44
xLoc=92
yLoc=73

xAx=np.linspace(0, refoc_mean_stack.shape[0], refoc_mean_stack.shape[0])  #x in um
################################
#     Synthetically Refocused       #
################################

result_refoc=refoc_mean_stack

inFocus=40

# synthetically refocused
xy_refoc=result_refoc[inFocus,:,xLoc:xLoc+1]
yx_refoc=result_refoc[inFocus,yLoc:yLoc+1,:]

xz_refoc=result_refoc[:,:,xLoc]
yz_refoc=result_refoc[:,yLoc,:]

y=9.5
lengthSB=10
xS=7.5

reconstruMeth='refoc'
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(result_refoc[41,yLoc-5:yLoc+6,xLoc-5:xLoc+6])
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
plt.savefig(figureFolder + r'\\FOV_{}_{}umSB.png'.format(reconstruMeth,lengthSB), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_{}_{}umSB.eps'.format(reconstruMeth,lengthSB), format='eps', dpi=800, bbox_inches='tight')

FOV=10
z1=10
z2=70

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(xz_refoc[z1:z2,round(yLoc-(FOV/2)):round(yLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_refocused_60um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_refocused_60um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(yz_refoc[z1:z2,round(xLoc-(FOV/2)):round(xLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_refocused_60um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_refocused_60um.eps', format='eps', dpi=800, bbox_inches='tight')


xLoc=60
yLoc=82

inFocus=40
result_decon=decon_mean_stack[1]
# synthetically refocused
xy=result_decon[inFocus,:,xLoc:xLoc+1]
yx=result_decon[inFocus,yLoc:yLoc+1,:]

xz=result_decon[:,:,xLoc]
yz=result_decon[:,yLoc,:]




fig = plt.gcf()      
ax = plt.subplot(111)  
#plt.imshow(yz[:,round(xLoc-(FOV/2)):round(xLoc+(FOV/2))])
plt.imshow(xz[:,round(yLoc-(FOV/2)):round(yLoc+(FOV/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_decon_80um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_decon_80um.eps', format='eps', dpi=800, bbox_inches='tight')




################################
#              Decon           #
################################
all_decon_660=decon_mean_stack

xy_dec=[]
yx_dec=[]
for ii in range(len(all_decon_660)):
    xy_dec.append(all_decon_660[ii,40,:,xLoc:xLoc+1])
    yx_dec.append(all_decon_660[ii,40,yLoc:yLoc+1,:])
    
xy_dec=np.array(xy_dec)
yx_dec=np.array(yx_dec)

xz_dec=[]
yz_dec=[]
for ii in range(len(all_decon_660)):
    xz_dec.append(all_decon_660[ii,:,:,xLoc])
    yz_dec.append(all_decon_660[ii,:,yLoc,:])
    
xz_dec=np.array(xz_dec)
yz_dec=np.array(yz_dec)



x1=35
x2=95
y1=25
y2=85
z1=10
z2=70



# VARIANCE FIGURES
# max projections
refoc_ROI= result_refoc[:,yLoc-10:yLoc+10,xLoc-10:xLoc+10]
refMaxProj_z = np.max(refoc_ROI, axis=0)
refMaxProj_x = np.max(refoc_ROI, axis=1)
refMaxProj_y = np.max(refoc_ROI, axis=2)



all_decon_660_ROI= all_decon_660[:,:,yLoc-10:yLoc+10,xLoc-10:xLoc+10]
decMaxProj_z = np.max(all_decon_660_ROI[1], axis=0)
decMaxProj_x = np.max(all_decon_660_ROI[1], axis=1)
decMaxProj_y = np.max(all_decon_660_ROI[1], axis=2)

xS=8
y=10
reconstruMeth='MaxProj_decon_660_3it'
reconstruMeth='MaxProj_ref'
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(decMaxProj_z[5:17,4:16])
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
plt.savefig(figureFolder + r'\\FOV_{}_{}umSB.png'.format(reconstruMeth,lengthSB), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_{}_{}umSB.eps'.format(reconstruMeth,lengthSB), format='eps', dpi=800, bbox_inches='tight')

FOV=20
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(decMaxProj_x[20:60,4:16])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_MaxProj_decon-3it_660_40um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_MaxProj_decon-3it_660_40um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(decMaxProj_y[20:60,5:17])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_MaxProj_decon-3it_660_40um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_MaxProj_decon-3it_660_40um.eps', format='eps', dpi=800, bbox_inches='tight')

# refocused
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(refMaxProj_x[20:60,4:16])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_MaxProj_ref_40um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_MaxProj_ref_40um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(refMaxProj_y[20:60,5:17])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_MaxProj_ref_40um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_MaxProj_ref_40um.eps', format='eps', dpi=800, bbox_inches='tight')





#########################
#     Lateral FWHM DECON & REF      #
#########################
FWHM_xy_all=np.zeros((8))
FWHM_yx_all=np.zeros((8))

spline = UnivariateSpline(xLat, xy_refoc-np.max(xy_refoc)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xy=r2-r1
spline = UnivariateSpline(xLat, yx_refoc-np.max(yx_refoc)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yx=r2-r1
fields=['refoc',FWHM_xy,FWHM_yx,xLoc,yLoc,figureFolder]
gf.appendCSV(analysisFolder ,r'\lateralResolution_ref_all-decon_FWHM',fields)

for ii in range(len(decon_mean_stack)): 
    print('decon',num_it[ii])
    y=xy_dec[ii,:,0]
    spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM_xy=r2-r1
    FWHM_xy_all[ii]=FWHM_xy
    
    y=yx_dec[ii,0,:]
    spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM_yx=r2-r1
    FWHM_yx_all[ii]=FWHM_yx
    
    fields=[num_it[ii],FWHM_xy,FWHM_yx,xLoc,yLoc,figureFolder]
    gf.appendCSV(analysisFolder ,r'\lateralResolution_ref_all-decon_FWHM',fields)


#########################
#      Axial FWHM       #
#########################
y=xz_refoc[:,yLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_xz=r2-r1

y=yz_refoc[:,xLoc]
spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
r1, r2 = spline.roots() # find the roots
FWHM_yz=r2-r1


fields=['refoc',FWHM_xz,FWHM_yz,xLoc,yLoc,figureFolder]
gf.appendCSV(analysisFolder ,r'\axialResolution_ref_decon-all_FWHM',fields)


xz=[]
yz=[]
for ii in range(len(decon_mean_stack)):
    xz.append(decon_mean_stack[ii,:,:,xLoc])
    yz.append(decon_mean_stack[ii,:,yLoc,:])
    
xz=np.array(xz)
yz=np.array(yz)

FWHM_xz_all=np.zeros((8))
FWHM_yz_all=np.zeros((8))
for ii in range(len(decon_mean_stack)): 
    print(num_it[ii])
    y=xz[ii,:,yLoc]    
    try:
        spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
        r1, r2 = spline.roots() # find the roots
        fit='N'
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
        fit='Y'
    FWHM_xz=r2-r1
    FWHM_xz_all[ii]=FWHM_xz
    
    y=yz[ii,:,xLoc]  
    try:
        spline = UnivariateSpline(xAx, y-np.max(y)/2, s=0)
        r1, r2 = spline.roots() # find the roots
        fit='N'
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
        fit='Y'
    FWHM_yz=r2-r1
    FWHM_yz_all[ii]= FWHM_yz
    print(FWHM_yz,fit)
    fields=[num_it[ii],FWHM_xz,FWHM_yz,xLoc,yLoc,figureFolder]
    gf.appendCSV(analysisFolder ,r'\axialResolution_ref_decon-all_FWHM',fields)



sum_550=np.sum(all_decon_550,axis=2)
sum_550=np.sum(sum_550,axis=2)   
sum_660=np.sum(all_decon_660,axis=2)
sum_660=np.sum(sum_660,axis=2)  

max_660=np.max(all_decon_660,axis=2)
max_660=np.max(max_660,axis=2) 
max_refoc=np.max(result_refoc,axis=1)
max_refoc=np.max(max_refoc,axis=1)  



# sum over a smaller area
sum_refoc_z=np.sum(refMaxProj_z,axis=0)
sum_refoc_x=np.sum(refMaxProj_x,axis=1)
sum_refoc_y=np.sum(refMaxProj_y,axis=1)

decMaxProj_z=np.zeros((8,20,20))
decMaxProj_x=np.zeros((8,81,20))
decMaxProj_y=np.zeros((8,81,20))
sum_dec_z=np.zeros((8,20))
sum_dec_x=np.zeros((8,81))
sum_dec_y=np.zeros((8,81))

for it in range(8):
    decMaxProj_z[it] = np.max(all_decon_660_ROI[it], axis=0)
    decMaxProj_x[it] = np.max(all_decon_660_ROI[it], axis=1)
    decMaxProj_y[it] = np.max(all_decon_660_ROI[it], axis=2)


for it in range(8):
    sum_dec_z[it]=np.sum(decMaxProj_z[it],axis=0)
    sum_dec_x[it]=np.sum(decMaxProj_x[it],axis=1)
    sum_dec_y[it]=np.sum(decMaxProj_y[it],axis=1)


# line plots instead 
line_ref_z=refMaxProj_z[10,:]
line_ref_x=refMaxProj_x[:,10]
line_ref_y=refMaxProj_y[:,10]

line_dec_z=decMaxProj_z[:,10,:]
line_dec_x=decMaxProj_x[:,:,10]
line_dec_y=decMaxProj_y[:,:,10]

plt.plot(np.transpose(line_dec_x))


color_idx = np.linspace(0, 1, 9)


fig = plt.gcf()    
ax = fig.add_subplot(211)
for col,ii in zip(color_idx, range(len(decon_mean_stack))):
    lines=plt.plot(xAx,sum_550[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(xAx_refoc,sum_refoc,linewidth=2.5,linestyle='--')
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=2)  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])

fig = plt.gcf()    
ax = fig.add_subplot(212)
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(xAx,sum_660[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(xAx_refoc,sum_refoc,linewidth=2.5,color='k',linestyle='--')
ax.set_xlabel('z (um)')    
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,7.5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\sumAllPixels_550-660_cm-tab10.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\sumAllPixels_550-660_cm-tab10.eps', format='eps', dpi=600, bbox_inches='tight')



# Lateral normalised
normMaxRefLat=[float(i)/np.max(xy_refoc) for i in xy_refoc]


normMaxDec660Lat=np.zeros((8,101))
for ii in range(len(xy_dec)):
    normMaxDec660Lat[ii]=[float(i)/np.max(xy_dec[ii]) for i in xy_dec[ii]]

test=np.transpose(yx_dec,[0,2,1])
normMaxDec660Lat=np.zeros((8,101))
for ii in range(len(yx_dec)):
    normMaxDec660Lat[ii]=[float(i)/np.max(test[ii]) for i in test[ii]]
    


fig = plt.gcf()    
ax = fig.add_subplot(111)
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(xLat-302,normMaxDec660Lat[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(xLat-368.5,normMaxRefLat,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
#fig.set_size_inches(7.5,7.5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\lateralNorm_660-refoc_cm-tab10.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\lateralNorm_660-refoc_cm-tab10.eps', format='eps', dpi=600, bbox_inches='tight')



# axial
normMaxRef=[float(i)/np.max(max_refoc) for i in max_refoc]

normMaxDec660=np.zeros((8,81))
for ii in range(len(max_660)):
    normMaxDec660[ii]=[float(i)/np.max(max_660[ii]) for i in max_660[ii]]


# new

normMaxRef=[float(i)/np.max(sum_refoc_x) for i in sum_refoc_x]
normMaxRef=(sum_refoc_x[18:60] - np.min(sum_refoc_x[18:60]))/(np.max(sum_refoc_x[18:60]) - np.min(sum_refoc_x[18:60]))


normMaxDec660=np.zeros((8,81))
for ii in range(len(line_dec_x)):
    normMaxDec660[ii]=[float(i)/np.max(sum_dec_x[ii]) for i in sum_dec_x[ii]]


fig = plt.gcf()    
ax = fig.add_subplot(111)
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(depths_dec+3,normMaxDec660[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(depths_ref[18:60]+2,normMaxRef,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
#fig.set_size_inches(7.5,7.5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\sumAllPixelsNorm_660-refoc_cm-tab10.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\sumAllPixelsNorm_660-refoc_cm-tab10.eps', format='eps', dpi=600, bbox_inches='tight')



fig = plt.gcf()    
ax = fig.add_subplot(211)
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(xLat-414,normMaxDec660Lat[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(xLat-368.5,normMaxRefLat,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])

ax = fig.add_subplot(212)
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(depths_dec+3,normMaxDec660[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(depths_ref[18:60]+2,normMaxRef,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
#fig.set_size_inches(7.5,7.5)
plt.tight_layout()  


fig.set_size_inches(6.5,6.5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\lateralAxialNorm_660-refoc_cm-tab10.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\lateralAxialNorm_660-refoc_cm-tab10.eps', format='eps', dpi=600, bbox_inches='tight')




fig = plt.gcf()    
ax = fig.add_subplot(211)
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(xLat,xy_dec[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(xLat,np.array(normMaxRefLat)/1000,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([310,370])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])

ax = fig.add_subplot(212)
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(depths_dec,normMaxDec660[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(depths_ref,normMaxRef,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
#fig.set_size_inches(7.5,7.5)
plt.tight_layout()  


fig.set_size_inches(6.5,6.5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\lateralAxialNorm_660-refoc_cm-tab10.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\lateralAxialNorm_660-refoc_cm-tab10.eps', format='eps', dpi=600, bbox_inches='tight')



# WIDEFIELD
inFocus_wf=131
x_wf=1127
y_wf=1880


xy_wf=wfStack[inFocus_wf,:,x_wf:x_wf+1]
yx_wf=wfStack[inFocus_wf,y_wf:y_wf+1,:]

xz_wf=wfStack[:,:,x_wf]
yz_wf=wfStack[:,y_wf,:]


conversion=2048/101

xS_wf=172
yS_wf=218

#5*conversion):round(17*conversion)
#round(4*conversion):round(16*conversion)
FOV_wf=round(12*conversion)

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(wfStack[inFocus_wf,round(y_wf-(FOV_wf/2)):round(y_wf+(FOV_wf/2)),round(x_wf-(FOV_wf/2)):round(x_wf+(FOV_wf/2))])
plt.plot([xS_wf,xS_wf+(lengthSB/0.26)],[yS_wf,yS_wf],linewidth=6.0,color='w')

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
pf.forceAspect(ax,aspect=1)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\FOV_wf_{}umSB.png'.format(lengthSB), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_wf_{}umSB.eps'.format(lengthSB), format='eps', dpi=800, bbox_inches='tight')



fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(wfStack[inFocus_wf-20:inFocus_wf+20,round(y_wf-(FOV_wf/2)):round(y_wf+(FOV_wf/2)),x_wf])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xz_wf_40um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xz_wf_40um.eps', format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(wfStack[inFocus_wf-20:inFocus_wf+20,y_wf,round(x_wf-(FOV_wf/2)):round(x_wf+(FOV_wf/2))])

#axis formatting
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
#ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([]) 
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_wf_40um.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_wf_40um.eps', format='eps', dpi=800, bbox_inches='tight')




norm_wfLat=[float(i)/np.max(wf_xProf[:,1]) for i in wf_xProf[:,1]]
norm_wfLat_y=[float(i)/np.max(wf_yProf[:,1]) for i in wf_yProf[:,1]]


wf_zProf_ROI=wf_zProf[wfProf_zCent-23:wfProf_zCent+17,1]
norm_wfAx=[float(i)/np.max(wf_zProf_ROI) for i in wf_zProf_ROI]

#alternate norm
norm_wfAx=(wf_zProf_ROI - np.min(wf_zProf_ROI))/(np.max(wf_zProf_ROI) - np.min(wf_zProf_ROI))

norm_decon_1iter=(normMaxDec660[0,17:63] - np.min(normMaxDec660[0,17:63]))/(np.max(normMaxDec660[0,17:63]) - np.min(normMaxDec660[0,17:63]))


wfProf_xCent=207
wfProf_zCent=130


plt.plot(xLat_wf,norm_wfLat[wfProf_xCent-77:wfProf_xCent+77])
plt.plot(norm_wfLat_y)



xLat_wf=np.arange(-20,20,0.26)
xAx_wf=np.arange(-20,20,1)

# widefield refocused decon
fig = plt.gcf()    
ax = fig.add_subplot(211)
plt.plot(xLat_wf,norm_wfLat[wfProf_xCent-78:wfProf_xCent+76],color='#551a8b',linewidth=2.5,linestyle=':')
for col,ii in zip(color_idx, range(len(all_decon_660))):
    lines=plt.plot(xLat-414,normMaxDec660Lat[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(xLat-368.5,normMaxRefLat,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])

ax = fig.add_subplot(212)
plt.plot(xAx_wf,norm_wfAx,color='#551a8b',linewidth=2.5,linestyle=':')
plt.plot(depths_dec[20:66],norm_decon_1iter,linewidth=2.5)         
for col,ii in zip(color_idx[1:9], range(7)):
    lines=plt.plot(depths_dec+3,normMaxDec660[ii+1,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.plot(depths_ref[18:60]+2,normMaxRef,linewidth=2.5,color='k',linestyle='--')
#plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=1)  
plt.xlim([-20,20])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
#fig.set_size_inches(7.5,7.5)
plt.tight_layout()  

fig.set_size_inches(6.5,6.5)
plt.tight_layout()  

plt.savefig(figureFolder + r'\\lateralAxialNorm_wf_660-refoc_cm-tab10.png', format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\lateralAxialNorm_wf_660-refoc_cm-tab10.eps', format='eps', dpi=600, bbox_inches='tight')





