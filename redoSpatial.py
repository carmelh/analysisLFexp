# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 16:42:02 2021

@author: chowe7
"""

# REDO SPATIAL

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


stack = tifffile.imread(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190730\slice2\cell1\z_stack\MLA3_1x1_50ms_150pA_1\2112_redo\Substack13.tif')


num_iterations=[1,3,5,7,9,13,17,21]

#r,center = (np.array([Rdy,Rdx]),np.array([df.at[currentFile, 'y'],df.at[currentFile, 'x']])) 

r,center = (np.array([0.09,19.52]),np.array([1021.1,1024])) 
depths = np.arange(-40,41,1)


pixelSize=5


refoc_mean_stack = get_refocussed(stack,r,center,depths,n_views = 19)
refoc_mean_stack= result


# 550 !!!!!!!!!!!
sum_ = np.sum(stack)
Nnum = 19
new_center = (1023,1023)
locs = de.get_locs(new_center,Nnum)

folder_to_data = 'H:/Python_Scripts/FireflyLightfield/PSF/correct_prop_550/'
df_path = r'H:\Python_Scripts\FireflyLightfield\PSF\correct_prop_550\sim_df.xlsx'

H = de.load_H_part(df_path,folder_to_data,zmax = 41*10**-6,zmin = -40*10**-6,zstep =1)

all_decon_550=np.zeros((8,len(H),101,101))
for it in range(len(num_iterations)):
    print(num_iterations[it])
    rectified = de.rectify_image(stack,r,center,new_center,Nnum)
    start_guess = de.backward_project(rectified/sum_,H,locs)
    result_rl = de.RL(start_guess,rectified/sum_,H,num_iterations[it],locs)
    all_decon_550[it] =result_rl   
    
    
# 660 !!!!!!
folder_to_data = 'Y:/home/psf_calculation/prop_660/'
df_path = r'Y:/home/psf_calculation/prop_660/sim_df.xlsx'
  
H = de.load_H_part(df_path,folder_to_data,zmax = 41*10**-6,zmin = -40*10**-6,zstep =1)
  
all_decon_660=np.zeros((8,len(H),101,101))
for it in range(len(num_iterations)):
    print(num_iterations[it])
    rectified = de.rectify_image(stack,r,center,new_center,Nnum)
    start_guess = de.backward_project(rectified/sum_,H,locs)
    result_rl = de.RL(start_guess,rectified/sum_,H,num_iterations[it],locs)
    all_decon_660[it] =result_rl   
    



figureFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ZSTCK\MLA_NOPIP_1x1_f_2-8_660nm_200mA_1\2201_redo\Figures'


################################
#     Synthetically Refocused       #
################################
#open files
#result=

result_refoc=refoc_mean_stack

xLat=np.linspace(0, result_refoc.shape[1]*pixelSize, result_refoc.shape[2])  #x in um
xAx=np.linspace(0, result_refoc.shape[0], result_refoc.shape[0])  #x in um
plt.imshow(result_refoc[40,...])

inFocus=40

# synthetically refocused
xy_refoc=result_refoc[inFocus,:,xLoc:xLoc+1]
yx_refoc=result_refoc[inFocus,yLoc:yLoc+1,:]


xz_refoc=result_refoc[:,:,xLoc]
yz_refoc=result_refoc[:,yLoc,:]



########################
#     DECONVOLVED      #
########################
num_it = [1,3,5,7,9,13,17,21]

#190605\slice1\Cell2
xLoc=43
yLoc=63

xLoc=51
yLoc=58

#bead
xLoc=51
yLoc=67

pixelSize=5

deconvolved_result=decon_mean_stack
reconstruMeth='decon_660'

xLat=np.linspace(0, deconvolved_result.shape[3]*pixelSize, deconvolved_result.shape[3])  #x in um
xAx=np.linspace(0, deconvolved_result.shape[1], deconvolved_result.shape[1])  #x in um


xy=[]
yx=[]
for ii in range(len(deconvolved_result)):
    xy.append(deconvolved_result[ii,40,:,xLoc:xLoc+1])
    yx.append(deconvolved_result[ii,40,yLoc:yLoc+1,:])
    
xy=np.array(xy)
yx=np.array(yx)


figureFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Phantoms\NonScattering\10umBeads_best\lf_refl660nm_stack_1\z18\figures'

# FOV image with scale bar
lengthSB = 25
xS=12.2
y=17.2

#for bead -7 +8
#y = 13
#xS=9

fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(result_refoc[40,yLoc-10:yLoc+10,xLoc-10:xLoc+10])
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
plt.savefig(figureFolder + r'\\FOV_refoc_{}umSB.png'.format(lengthSB), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_refoc_{}umSB.eps'.format(lengthSB), format='eps', dpi=800, bbox_inches='tight')


fig = plt.gcf()      
ax = plt.subplot(111)  
plt.imshow(deconvolved_result[1,40,yLoc-10:yLoc+10,xLoc-10:xLoc+10])
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
plt.savefig(figureFolder + r'\\FOV_decon_3it_{}umSB.png'.format(lengthSB), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\FOV_decon_3it_{}umSB.eps'.format(lengthSB), format='eps', dpi=800, bbox_inches='tight')


date='190605'
analysisFolder=r'Y:\projects\thefarm2\live\Firefly\Lightfield\Phantoms\NonScattering\10umBeads_best\lf_refl660nm_stack_1\z18'


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
    y=xy[ii,:,0]
    spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM_xy=r2-r1
    FWHM_xy_all[ii]=FWHM_xy
    
    y=yx[ii,0,:]
    spline = UnivariateSpline(xLat, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    FWHM_yx=r2-r1
    FWHM_yx_all[ii]=FWHM_yx
    
    fields=[num_it[ii],FWHM_xy,FWHM_yx,xLoc,yLoc,figureFolder]
    gf.appendCSV(analysisFolder ,r'\lateralResolution_ref_all-decon_FWHM',fields)



color_idx = np.linspace(0, 1, 9)

fig = plt.gcf()    
ax = fig.add_subplot(111)
for col,ii in zip(color_idx, range(len(deconvolved_result))):
    lines=plt.plot(xLat,xy[ii,:,0],linewidth=2.5, color=plt.cm.tab10(col))
    
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
ax.set_xlabel('x (um)')    
ax.set_xlim((300,380))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\xy_{}_cm-tab10.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\xy_{}_cm-tab10.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')


fig = plt.gcf()    
ax = fig.add_subplot(111)
for col,ii in zip(color_idx, range(len(deconvolved_result))):
    lines=plt.plot(xLat,yx[ii,0,:],linewidth=2.5, color=plt.cm.tab10(col))
    
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
ax.set_xlabel('x (um)')    
ax.set_xlim((220,300))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yx_{}_cm-tab10.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yx_{}_cm-tab10.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')




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


fields=['refoc',FWHM_xz*z_spacing,FWHM_yz*z_spacing,xLoc,yLoc,figureFolder]
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
    
    
    
fig = plt.gcf()    
ax = fig.add_subplot(111)
for col,ii in zip(color_idx, range(len(deconvolved_result))):
    lines=plt.plot(xAx,yz[ii,:,xLoc],linewidth=2.5, color=plt.cm.tab10(col))
    
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False)
ax.set_xlabel('z (um)')    
ax.set_xlim((20,50))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
fig.set_size_inches(7.5,5)
plt.tight_layout()  
plt.savefig(figureFolder + r'\\yz_{}_cm-tab10.png'.format(reconstruMeth), format='png', dpi=600, bbox_inches='tight')
plt.savefig(figureFolder + r'\\yz_{}_cm-tab10.eps'.format(reconstruMeth), format='eps', dpi=600, bbox_inches='tight')

    


avg_550=np.average(all_decon_550,axis=2)
avg_550=np.average(avg_550,axis=2)   

sum_550=np.sum(all_decon_550,axis=2)
sum_550=np.sum(sum_550,axis=2)   
sum_660=np.sum(all_decon_660,axis=2)
sum_660=np.sum(sum_660,axis=2)  

max_550=np.max(all_decon_550,axis=2)
max_550=np.max(max_550,axis=2)   
max_660=np.max(all_decon_660,axis=2)
max_660=np.max(max_660,axis=2) 

fig = plt.gcf()    
ax = fig.add_subplot(211)
for col,ii in zip(color_idx, range(len(deconvolved_result))):
    lines=plt.plot(xAx,sum_550[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
plt.legend(['1', '3', '5', '7', '9', '13', '17', '21'],frameon=False,ncol=2)  
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])

ax = fig.add_subplot(212)
for col,ii in zip(color_idx, range(len(deconvolved_result))):
    lines=plt.plot(xAx,sum_660[ii,:],linewidth=2.5, color=plt.cm.tab10(col))
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
