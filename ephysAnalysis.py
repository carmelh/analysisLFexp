# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:30:21 2019

@author: chowe7
"""

import neo
import sys
import quantities as pq
import numpy as np
import matplotlib.pyplot as plt


def load_ephys(path,lazy = False):
    reader = neo.io.Spike2IO(path)
    bl = reader.read(cascade = True,lazy = lazy)[0]
    
    return bl


def get_ephys_datetime(file_path):
    '''
    See header description below
    Reads the correct bytes from the ephys header for the 'datetime_year' and 'datetime_detail' fields
    '''
    year = np.fromfile(file_path,dtype = np.uint16,count = 30)[29]
    #ucHun,ucSec,ucMin,ucHour,ucDay,ucMon = np.fromfile(file_path,dtype = np.uint8,count = 59)[52:58].astype(str)
    detail = np.fromfile(file_path,dtype = np.uint8,count = 59)[52:58].astype(str)
    time = detail[1:4]
    time2 = []
    for val in time[::-1]:
        if len(val) == 2:
            time2.append(val)
        else:
            time2.append(str(0)+val)
        
    
    day = int(str(year)+str(detail[-1])+str(detail[-2]))
    time = int(time2[0]+time2[1]+time2[2])
    ms = str(int(detail[0])*100)
    return day,time,ms


def getChannels(block):
    #LED = block.segments[0].analogsignals[2]    
    current = block.segments[0].analogsignals[1]
    voltage = block.segments[0].analogsignals[0]
    times = block.segments[0].analogsignals[1].times
       # times = block.segments[0].analogsignals[1].times.rescale('s').magnitude


    return current, voltage, times


def getKeyboardEntries(block):
    # Keyboard commands are stored in ASCII form- this function coverts them to CHARs
    
    keyboardChannel = block.segments[0].events[1]   #this channel holds the keyboard commands
    eventsASCII = keyboardChannel.labels    # pulls out the keyboard entries (which are in ASCII format by default)
    eventTimes = keyboardChannel.times      # pulls out the times for each key press
    
    keyboardEntries = []
    for element in eventsASCII:
            keyboardEntries.append(chr(int(element))) # converts ASCII to char
            
    return keyboardEntries, eventTimes   


def time_to_idx(analogSignal,time):

    
    try:
        idx = int(round((time - analogSignal.t_start)*analogSignal.sampling_rate))
   # except Exception:
    #    idx = int(round((time*pq.s - analogSignal.t_start)*analogSignal.sampling_rate))
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise
    return idx
    
            
def trialSplitter(vcCurrentTrace, stimulusOnsetIndices):
    # This function takes the stimulation times, and uses them to segment the current traces 
    # The time before the stim is taken as an average. the time after is taken to discern the photocurrent.
    
    allTrials = []
    
    for element in stimulusOnsetIndices:
        allTrials.append(vcCurrentTrace[element-preStimLength:element+postStimLength])
    
    return allTrials


#-------------------------------------------------------#
#                                                       #
#-------------------------------------------------------#
 

ephyspath = r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ephys1.smr'

ephys = load_ephys(ephyspath, False)
[day,time,ms] = get_ephys_datetime(ephyspath)
current, voltage, times = getChannels(ephys)
kEnt, eventTimes = getKeyboardEntries(ephys)

kEnt[22]


#fs = ephys.segments[0].analogsignals[0].sampling_rate
#ts=ephys.segments[0].analogsignals[0].t_start


#idx = round((times - ts)*fs).rescale('s*Hz').magnitude

times2=times.rescale('s').magnitude
stimTime = eventTimes[22].rescale('s').magnitude
stimEnd = eventTimes[22].rescale('s').magnitude + 10

idxs=np.where((times2 >= stimTime)&(times2 <= stimEnd))

idxStart=[lis[0] for lis in idxs]
idxEnd=[lis[len(idxs[0])-1] for lis in idxs]

#idxStart=times2[41495587]
#idxStart=times2[41897192]

timesNew= times2[idxStart:idxEnd]
timesNew = np.arange(0, 1/fs.rescale('Hz').magnitude*len(timesNew), 1/fs.rescale('Hz').magnitude)


#-------------------------------------------------------#
#                      PLOT                             #
#-------------------------------------------------------#


font = {'family': 'sans',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

## voltage
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(timesNew,voltage[41495587:41897192],linewidth=3.0,color='k')
plt.plot([8,8],[-20,0],linewidth=4.0,color='k')
plt.plot([8,10],[-20,-20],linewidth=4.0,color='k')
fig.set_size_inches(8,4)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([])
plt.savefig(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4\voltageEphys.eps', format='eps', dpi=1000)
    
## current
fig = plt.gcf()      
ax = plt.subplot(111)  
plt.plot(timesNew,current[idxStart:idxEnd],linewidth=3.0,color='k')
plt.plot([8,8],[0.3,0.4],linewidth=4.0,color='k')
plt.plot([8,10],[0.3,0.3],linewidth=4.0,color='k')
fig.set_size_inches(8,2)
    
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_linewidth(False)
ax.spines['left'].set_linewidth(False)
ax.axes.get_yaxis().set_ticks([])
ax.axes.get_xaxis().set_ticks([])
plt.savefig(r'Y:\projects\thefarm2\live\Firefly\Lightfield\Calcium\CaSiR-1\Intra\190605\slice1\Cell2\ACT-MLA_1x1_f_2-8_50ms_660nm_200mA-ASTIM_4\currentEphys.eps', format='eps', dpi=1000)
    


