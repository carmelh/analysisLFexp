# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:45:42 2021

@author: chowe7
"""

from scipy import signal

plt.imshow(np.var(delta_f[:,100:400,100:400],axis=-0))


plt.imshow(np.var(result,axis=-0))





x,trialData = gf.importCSV(r'Y:\projects\thefarm2\live\Firefly\NIR-GECO_imaging\210325\slice1\area1\s1a1_WF_1P_1x1_200mA_100msExp_func_600frames_3','WF_1P_1x1_200mA_100msExp_func_600frames_3_MMStack_Default.ome')



trialData = np.average(delta_f[:,302:317,297:310], axis=1)    
trialData = np.average(trialData,axis=1)


trialData1 = np.average(delta_f[:,207:222,243:256], axis=1)    
trialData1 = np.average(trialData1,axis=1)


    ts=100e-3
    time = np.arange(0, ts*len(trialData), ts)

    fig = plt.gcf()      
    ax = plt.subplot(111)  
    plt.plot(time,signal.detrend(trialData),linewidth=3.0,color='k')
    plt.plot([5,5],[-8,-13],linewidth=4.0,color='k')
    plt.plot([5,10],[-13,-13],linewidth=4.0,color='k')
    fig.set_size_inches(8,4)
    
    #axis formatting
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(False)
    ax.spines['left'].set_linewidth(False)
    ax.axes.get_yaxis().set_ticks([])
    ax.axes.get_xaxis().set_ticks([]) 
    plt.savefig(r'Y:\projects\thefarm2\live\Firefly\NIR-GECO_imaging\210325\slice1\area1\s1a1_WF_1P_1x1_200mA_100msExp_func_600frames_3\Figures\timeSeries_5sec_2per.png', format='png', dpi=600, bbox_inches='tight')

