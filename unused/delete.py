# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:46:44 2020

@author: chowe7
"""

pd_dec = pd.read_csv(cwd + r'\stats_deconvolved_reg_{}.csv'.format(date))        



diff=0
fig = plt.figure(figsize=(15, 8))    
ax1 = plt.subplot(231)
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['peakF'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['peakF'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['peakF'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
plt.legend(frameon=False)
pf.lrBorders(ax1)
plt.ylabel('Peak Signal (%)', fontdict = font)

ax2 = plt.subplot(232)
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['dfNoise'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['dfNoise'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['dfNoise'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax2)
plt.ylabel('Noise (%)', fontdict = font)hmm 
plt.xlabel('Iteration', fontdict = font)

ax3 = plt.subplot(233)
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['SNR'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['SNR'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['SNR'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax3)
plt.ylabel('SNR (%)', fontdict = font)

diff=48
ax4 = plt.subplot(234)
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['peakF'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['peakF'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['peakF'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax4)
plt.ylabel('Peak Signal (%)', fontdict = font)

ax5 = plt.subplot(235)
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['dfNoise'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['dfNoise'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['dfNoise'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax5)
plt.ylabel('Noise (%)', fontdict = font)
plt.xlabel('Iteration', fontdict = font)

ax6 = plt.subplot(236)
plt.plot(pd_dec['numIt'][idx[0]+diff:idx[1]+diff],pd_dec['SNR'][idx[0]+diff:idx[1]+diff],linewidth=3.0,color='k',marker='.',markersize='20',label='RL')
plt.plot(pd_dec['numIt'][idx[2]+diff:idx[3]+diff],pd_dec['SNR'][idx[2]+diff:idx[3]+diff],linewidth=3.0,color='r',marker='.',markersize='20',label='Reg')
plt.plot(pd_dec['numIt'][idx[4]+diff:idx[5]+diff],pd_dec['SNR'][idx[4]+diff:idx[5]+diff],linewidth=3.0,color='b',marker='.',markersize='20',label='Reg 2')
pf.lrBorders(ax6)
plt.ylabel('SNR (%)', fontdict = font)
plt.tight_layout() 
pf.saveFigure(fig,figurePath,'\\tutto_90per')