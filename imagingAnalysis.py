#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 19:14:20 2018

@author: jeevan
@heavily edited: Carmel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy
import pandas as pd
import csv
import os

# This code was designed to read in the data seleceted for stats analyis.
# we want to report the average and range of: dF/F, SNR, baseline
# we want these to be in counts.


# This section creates the relevent LED-on markers to plot on the data
# Imaging at 100Hz 
stimIndices_0_5Hz = [89,289,489]
stimIndices_1_0Hz = [89,189,289]
stimIndices_2_0Hz = [89,139,189]
stimIndices_5_0Hz = [89,109,129]
stimIndices_10_0Hz = [89,99,109]
stimIndices_15_0Hz = [89,96,103]
stimIndices_20_0Hz = [89,94,99]

#------------------------------#
#          Functions           #
#------------------------------#


def getStimIndices(stimFrequency):
    if stimFrequency[0] == 0.5:
        return stimIndices_0_5Hz
    elif stimFrequency[0] == 1.0:
        return stimIndices_1_0Hz
    elif stimFrequency[0] == 2.0:
        return stimIndices_2_0Hz
    elif stimFrequency[0] == 5.0:
        return stimIndices_5_0Hz
    elif stimFrequency[0] == 10.0:
        return stimIndices_10_0Hz
    elif stimFrequency[0] == 15.0:
        return stimIndices_15_0Hz
    elif stimFrequency[0] == 20.0:
        return stimIndices_20_0Hz
    
    

def getStimArtefacts(darkTrial, stimIndices):
    # This looks at the max value of when the stim is on +/- 1 frame, and uses this to minus off the real data

    artefactMagnitude = []
    
    for index in stimIndices:
        peakValue = max(darkTrial[index-1:index+1])
        artefactMagnitude.append(peakValue-np.mean(darkTrial))
    
    return artefactMagnitude



def processRawTrace(trialData, darkTrial, backgroundTrace, baselineIdx):
    # Calculate the baseline Fluorescence at the beginning of each trace. 
    # Important: this doesn't take in to account photobleaching of the dye. 
    # The first xx frames are before any photostim
    
    baselineFluorescence = np.mean(trialData[0:baselineIdx])
    baselineBackgroundFluorescence = np.mean(backgroundTrace[0:baselineIdx])
    
    # calculate the average number of counts for the dark trial. This is to get the f_dark value for the dF/F calculation
    darkTrialAverage = np.mean(darkTrial)
    
    # Initialise the container for the processed traces
    processedTrace = []
    processedBackgroundTrace = []
    diff = []
    
    # now calc (f-f0)/(f0-fdark)      
    for element in trialData:
        processedTrace.append((element-baselineFluorescence)/(baselineFluorescence-darkTrialAverage))
        
    # now calc (f-f0)/(f0-fdark) for background trace    
    for element in backgroundTrace:
        processedBackgroundTrace.append((element-baselineBackgroundFluorescence)/(baselineBackgroundFluorescence-darkTrialAverage))
        
    diff =  np.array(processedTrace) - np.array(processedBackgroundTrace) # change to processedTrace to get stats for diff
    
    return processedTrace, diff, processedBackgroundTrace

 

def processRawTraceStimArtefacts(trialData, trialArtefacts, stimIndices, darkTrial, backgroundTrace, baselineIdx):
    # Calculate the baseline Fluorescence at the beginning of each trace. 
    # Important: this doesn't take in to account photobleaching of the dye. 
    # The first xx frames are before any photostim
    
    baselineFluorescence = np.mean(trialData[0:baselineIdx])
    baselineBackgroundFluorescence = np.mean(backgroundTrace[0:baselineIdx])
    
    # calculate the average number of counts for the dark trial. This is to get the f_dark value for the dF/F calculation
    darkTrialAverage = np.mean(darkTrial)
    
    # The following loop removes the stimulation artefacts
    for index, artefact in enumerate(trialArtefacts):
        trialData[int(stimIndices[index])] = trialData[int(stimIndices[index])]- artefact  
    
    # Initialise the container for the processed traces
    processedTrace = []
    processedBackgroundTrace = []
    diff = []
    
    # now calc (f-f0)/(f0-fdark)    
    for element in trialData:
        processedTrace.append((element-baselineFluorescence)/(baselineFluorescence-darkTrialAverage))
        
    # now calc (f-f0)/(f0-fdark) for background trace   
    for element in backgroundTrace:
        processedBackgroundTrace.append((element-baselineBackgroundFluorescence)/(baselineBackgroundFluorescence-darkTrialAverage))
        
    diff =  np.array(processedTrace) - np.array(processedBackgroundTrace) # change to processedTrace to get stats for diff
    
    return processedTrace, trialData, diff, processedBackgroundTrace

    
def getStatistics(processedTrace,trialData,darkTrialData,baselineIdx):    
    baseline = np.mean(trialData[0:baselineIdx])
    baseline_photons = baseline*100*2**16/30000
    baselineNoise = np.sqrt(np.var(trialData[0:baselineIdx]))
        
    maxValue = max(trialData[12:30])
    peakIdx = np.array(np.where(trialData == maxValue))
    peakSignal = np.mean(trialData[peakIdx[0,0]-2:peakIdx[0,0]+2])
    peakSignal_photons = peakSignal*100*2**16/30000
    
    peak_dF_F = (processedTrace[peakIdx[0,0]])*100 #in %
    
    df_noise = (np.sqrt(np.var(processedTrace[0:baselineIdx])))*100 # in %
    
    SNR = (peak_dF_F)/df_noise
    
    baselineDarkNoise = np.sqrt(np.var(darkTrialData))

    #get bleach rate
    total_bleach = np.mean(trialData[:10])-np.mean(trialData[-10:])
    bleach_df_percent_per_sec = -100*((total_bleach)/(baseline - np.mean(darkTrialData)))*(100/len(trialData))

  #  container_peak_signal.append(peakSignal-baseline)
    
    return baseline, baseline_photons, baselineNoise, peakSignal, peakSignal_photons, peak_dF_F, df_noise, SNR, baselineDarkNoise, bleach_df_percent_per_sec

   
    

    