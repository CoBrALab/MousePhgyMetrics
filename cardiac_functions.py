#!/usr/bin/env python
# coding: utf-8

""" Purpose: extract metrics from the raw plethysmography trace.
The final list of metrics to be obtained are:
(Instantaneous - over 2s) rate, period, HRV - std of period, HRV-RMSSD, HRV- wavelet % above halfmax, predictability-entropy, PVI,
            shape-FWQM, shape-FWHM, shape-FWB
(Within large time window) mean rate, std of rate across window, mean period, HRV - std of period across window, predictability - entropy across window, PVI across window?
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks, peak_widths
import warnings
import EntropyHub as EH
import gc

def repeat_values_for_plotting(data_to_repeat, beats_bool, beat_indices):
    '''For metrics where there is only 1 value per breath - duplicate the value until the next breath to create array
    with the length of all samples in original data - that way it can be plotted on the same graph as original trace.'''
    
    full_length_array = pd.Series(np.repeat(np.nan, len(beats_bool)))
    
    for i in range(0, len(beat_indices)-1):
        full_length_array[beat_indices[i]:beat_indices[i+1]] = np.repeat(data_to_repeat[i], beat_indices[i+1]-beat_indices[i])
        
    return full_length_array  

#function to denoise and detrend the data
def denoise(pulseox_trace, sampling_rate, invert_bool):
    #convert the input array to a pandas Series, since the rolling window function is a pandas function
    raw_trace = pd.Series(pulseox_trace)
    
    #smooth the raw trace by taking the mean within a rolling window of 20 samples
    #using the Gaussian weighting makes every element in the smoothed series unique with higher precision
    trace_smoothed = raw_trace.rolling(20, center = True, min_periods = 1, win_type = 'gaussian').mean(std=5)
    
    if invert_bool == True:
        trace_smoothed = (-1)*trace_smoothed
    
    return trace_smoothed

def find_heart_beats(trace_smoothed, height_val, dist_val, prominence_val, plat_max, width_val):
    
    #find most prominent peaks - setting a min width prevents detection of a bifurcated peak as two separate peaks
    beat_indices, beat_properties = find_peaks(trace_smoothed, height = height_val, distance = dist_val,
                                               prominence = prominence_val, plateau_size = (1,plat_max), width = width_val, rel_height = 1)
    
    #create boolean beat array
    beats_bool = pd.Series(np.repeat(0, len(trace_smoothed)))
    beats_bool[beat_indices] = 1
    
    #create array for plotting each breath on top of the original trace
    beats_toplot = (beats_bool*trace_smoothed).replace(0,np.nan)
        
    return beat_indices, beats_bool, beats_toplot, pd.Series(beat_properties['prominences'])

def get_HR_inst(beats, censoring_arr_full, window_length, sampling_rate):
    '''function to calculate instantaneous HR (based on # of beats in a rolling time window of a few sec)'''
    
    #calculate the number of beats in a time window of the specified length
    beat_sum = beats.rolling(window_length*sampling_rate, center = True).sum()

    #convert the sum of beats to HR in beats/min
    HR = (60/window_length)*beat_sum

    #smooth the HR trace such that there are fewer bumps when the rolling window encounters a new beat
    HR_smooth = HR.rolling(sampling_rate, center = True, win_type = 'gaussian').mean(std=100)
    
    #censor the necessary samples by setting them to nan
    HR_smooth_censored = np.copy(HR_smooth)
    HR_smooth_censored[censoring_arr_full] = np.nan
        
    return HR_smooth, HR_smooth_censored

def get_period(beats_bool, sampling_rate):
    #from the boolean beat array, extract indices of each beat
    beat_indices = np.where(beats_bool == 1)[0]
    
    #find the difference between breaths (ie between consecutive elements of the location array), divide by sampling_rate to get seconds
    period_btw_beats = pd.Series(beat_indices).diff(periods=1)/sampling_rate
    
    #find successive differences between consecutive periods and square
    period_ssd = pd.Series(period_btw_beats).diff(periods=1)**2
        
    return period_btw_beats, period_ssd

def get_inst_hrv(period_btw_beats, period_ssd, window_size):
    '''function to extract the heart rate variability (ie a metric of how much the HR changes over time) using the standard
    deviation of HR in a short rolling window and root mean square of successive differences (RMSSD) between 
    ALL consecutive breaths.'''
    
    #compute std of period btw breaths in a window of n breaths
    hrv_std_period = period_btw_beats.rolling(window_size, min_periods = 0, center = True).std()
    
    #compute RMSSD (root mean square of successive differences in period) by finding mean (then root) in a window of n breaths
    #rrv_rmssd_period = pd.Series(np.repeat(np.nan, len(period_btw_breaths)))
    hrv_rmssd_period = period_ssd.rolling(window_size-1, min_periods = 0, center = True).mean()**(1/2)
    
    return hrv_std_period, hrv_rmssd_period

def get_PVI(beat_prominence, num_beats):
    '''function to extract variability in the height of the pulse oximetry peaks - this metric is referred to as plethysmography 
    variability index (PVI) in the literature. I use the beat prominence because it gives the distance between the peak and through.
    It is typically computed over 8s (for humans). Here, we use rolling window (~2s) since the mouse heart rate is faster than that of a 
    human'''
    
    max_amp_in_window = beat_prominence.rolling(num_beats, min_periods=1, center = True).max().reset_index(drop=True)
    min_amp_in_window = beat_prominence.rolling(num_beats, min_periods=1, center = True).min().reset_index(drop=True)
    pvi_percent_change = 100*(max_amp_in_window - min_amp_in_window)/max_amp_in_window
    
    return pvi_percent_change

def get_pulse_shape_metrics(trace_smooth, beat_indices, sampling_rate):
    
    #get the peak widths (at 1/4, 1/2 and full height)
    peak_quarterwidths = peak_widths(trace_smooth, beat_indices, rel_height = 0.25) 
    peak_halfwidths = peak_widths(trace_smooth, beat_indices, rel_height = 0.5) 
    peak_fullwidths = peak_widths(trace_smooth, beat_indices, rel_height = 1) 
    
    #convert to time
    peak_quarterwidths_time = peak_quarterwidths[0]/sampling_rate
    peak_halfwidths_time = peak_halfwidths[0]/sampling_rate
    peak_fullwidths_time = peak_fullwidths[0]/sampling_rate
    
    return peak_quarterwidths_time, peak_halfwidths_time, peak_fullwidths_time

def get_periodiocity_wavelet(trace_smoothed, censoring_arr_full, sampling_rate, time_array,tot_num_samples, output_name, 
                             wavelet_band_width, wavelet_peak_dist, num_bands_to_detect, num_to_avg):
    '''this function computes the wavelet transform (frequency spectrum at each point in time - very robust to noise) then 
    examines how power is concentrated across the frequencies'''
 
    # pad the trace to reduce edge effects
    pad_samples = 500
    padded_trace = np.pad(trace_smoothed, pad_width = pad_samples, mode = 'reflect')
    
    #find wavelet transform with a morlet wavelet of width 8
    freq = np.linspace(1, sampling_rate/2, 1000)
    widths = (8.0)*sampling_rate / (2*freq*np.pi)
    cwtmatr = np.abs(signal.cwt(padded_trace, signal.morlet2, widths, w = 8.)[0:50,pad_samples:tot_num_samples+pad_samples])
    
    #normalize columns to max height to 1
    cwtmatr_norm_height = cwtmatr/np.max(cwtmatr) 

    #calculate the % that are located above the half max (gives approx measure of spread) and the num of distinct peaks
    periodicity_percent_spectrum_above_halfmax = pd.Series(np.repeat(np.nan, tot_num_samples))
    wavelet_peak_num = pd.Series(np.repeat(np.nan, tot_num_samples))
    wavelet_lowfreq_band = pd.Series(np.repeat(np.nan, tot_num_samples))
    wavelet_highfreq_band = pd.Series(np.repeat(np.nan, tot_num_samples))
    wavelet_vhighfreq_band = pd.Series(np.repeat(np.nan, tot_num_samples))
    wavelet_vhighfreq_band_height = pd.Series(np.repeat(np.nan, tot_num_samples))
    wavelet_highfreq_band_height = pd.Series(np.repeat(np.nan, tot_num_samples))
    wavelet_lowfreq_band_height = pd.Series(np.repeat(np.nan, tot_num_samples))
    halfmax = np.max(np.abs(cwtmatr), axis = 0)/2
    for col in range(0,tot_num_samples):
        indices_above_halfmax = np.where(cwtmatr[:,col] >= halfmax[col])[0]
        periodicity_percent_spectrum_above_halfmax[col] = 100*len(indices_above_halfmax)/50 #50 points per spectrum
        #smoothed_spectrum_cross_sec = pd.Series(cwtmatr_norm_height[:,col]).rolling(4, min_periods=0, center=True).mean()
        
        #get the number of distinct peaks (frequencies)
        peak_indices, peak_properties = find_peaks(cwtmatr_norm_height[:,col], height = 0.3, plateau_size = (1,5), distance = 
                                                   wavelet_peak_dist,prominence = 0)
        wavelet_peak_num[col] = len(peak_indices)
        sorted_heights = sorted(peak_properties['peak_heights'])[::-1]
        
        #if there is only one strong frequency
        try:
            pos_of_strongest_peak = np.where(np.array(peak_properties['peak_heights']) == sorted_heights[0])[0][0]
            freq_of_strongest_peak = freq[0:50][peak_indices[pos_of_strongest_peak]]
            #detect how similar the new freq peak is to prev frequencies in low/high frequency bands to place it in appropriate band
            if col == 0:
                wavelet_lowfreq_band[col] = freq_of_strongest_peak
                wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_strongest_peak]]
                wavelet_highfreq_band[col] = np.nan
            elif (prev_lowfreq + wavelet_band_width > freq_of_strongest_peak) and (prev_lowfreq - wavelet_band_width < freq_of_strongest_peak):
                wavelet_lowfreq_band[col] = freq_of_strongest_peak
                wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_strongest_peak]]
                wavelet_highfreq_band[col] = np.nan
            elif (prev_highfreq + wavelet_band_width > freq_of_strongest_peak) and (prev_highfreq - wavelet_band_width < freq_of_strongest_peak):
                wavelet_highfreq_band[col] = freq_of_strongest_peak
                wavelet_highfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_strongest_peak]]
                wavelet_lowfreq_band[col] = np.nan
            elif (prev_vhighfreq + wavelet_band_width > freq_of_strongest_peak) and (prev_vhighfreq - wavelet_band_width < freq_of_strongest_peak):
                wavelet_vhighfreq_band[col] = freq_of_strongest_peak
                wavelet_vhighfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_strongest_peak]]
            else:
                #print('Strongest freq cannot be classified, sample: ' + str(col) + '. It is ' + str(freq_of_strongest_peak) + ' - whereas prev low is ' + str(prev_lowfreq) + ' and prev high is ' + str (prev_highfreq))
                wavelet_lowfreq_band[col] = np.nan
                wavelet_highfreq_band[col] = np.nan
        except IndexError as error:
                wavelet_lowfreq_band[col] = np.nan
                wavelet_highfreq_band[col] = np.nan
        #if there is also a second strong frequency
        try:
            pos_of_secondary_peak = np.where(np.array(peak_properties['peak_heights']) == sorted_heights[1])[0][0]
            freq_of_secondary_peak = freq[0:50][peak_indices[pos_of_secondary_peak]]
            
            if col == 0:
                if freq_of_secondary_peak > freq_of_strongest_peak:
                    #if the second freq is higher than the first one
                    wavelet_lowfreq_band[col] = freq_of_strongest_peak
                    wavelet_highfreq_band[col] = freq_of_secondary_peak
                    wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_strongest_peak]]
                    wavelet_highfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_secondary_peak]]
                else:
                    wavelet_highfreq_band[col] = freq_of_strongest_peak
                    wavelet_lowfreq_band[col] = freq_of_secondary_peak
                    wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_secondary_peak]]
                    wavelet_highfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_strongest_peak]]
                
            elif np.isnan(prev_highfreq):
                if freq_of_secondary_peak > freq_of_strongest_peak:
                    wavelet_lowfreq_band[col] = freq_of_strongest_peak
                    wavelet_highfreq_band[col] = freq_of_secondary_peak
                    wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_strongest_peak]]
                    wavelet_highfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_secondary_peak]]
                else:
                    tmp_oldhigh = np.copy(wavelet_highfreq_band)
                    tmp_oldlow = np.copy(wavelet_lowfreq_band)
                    wavelet_highfreq_band = tmp_oldlow
                    wavelet_lowfreq_band = tmp_oldhigh
                    wavelet_lowfreq_band[col] = freq_of_secondary_peak
                    
                    tmp_oldhigh_height = np.copy(wavelet_highfreq_band_height)
                    tmp_oldlow_height = np.copy(wavelet_lowfreq_band_height)
                    wavelet_highfreq_band_height = tmp_oldlow_height
                    wavelet_lowfreq_band_height = tmp_oldhigh_height
                    print('SWITCH LOW AND HIGH')
                
            elif (freq_of_strongest_peak + wavelet_band_width > freq_of_secondary_peak) and (freq_of_strongest_peak - wavelet_band_width < freq_of_secondary_peak):
                print('Cannot distinguish well between strong and secondary frequencies, sample: ' + str(col))
            elif (prev_lowfreq + wavelet_band_width > freq_of_secondary_peak) and (prev_lowfreq - wavelet_band_width < freq_of_secondary_peak):
                wavelet_lowfreq_band[col] = freq_of_secondary_peak
                wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_secondary_peak]]
            elif (prev_highfreq + wavelet_band_width > freq_of_secondary_peak) and (prev_highfreq - wavelet_band_width < freq_of_secondary_peak):
                wavelet_highfreq_band[col] = freq_of_secondary_peak
                wavelet_highfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_secondary_peak]]
            elif (prev_vhighfreq + wavelet_band_width > freq_of_secondary_peak) and (prev_vhighfreq - wavelet_band_width < freq_of_secondary_peak):
                wavelet_vhighfreq_band[col] = freq_of_secondary_peak
                wavelet_vhighfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_secondary_peak]]
            else:
                #print('Secondary freq cannot be classified, sample: ' + str(col) + '. It is ' + str(freq_of_secondary_peak) + ' - whereas prev low is ' + str(prev_lowfreq) + ' and prev high is ' + str (prev_highfreq))
                tmp=0
        except IndexError as error:
            #only set previous value to nan if there have not yet been any values in this frequency band
            if col==0 or np.isnan(prev_highfreq):
                prev_highfreq = np.nan
                
        #if I know that I also want to look for a third frequency
        if num_bands_to_detect==3:
            try:
                pos_of_third_peak = np.where(np.array(peak_properties['peak_heights']) == sorted_heights[2])[0][0]
                freq_of_third_peak = freq[0:50][peak_indices[pos_of_third_peak]]

                if (col == 0) or np.isnan(prev_vhighfreq):
                    #assign the newfound third frequency to the right frequence band
                    #if the third freq is the highest
                    if (freq_of_third_peak > freq_of_strongest_peak) and (freq_of_third_peak > freq_of_secondary_peak):
                        wavelet_vhighfreq_band[col] = freq_of_third_peak
                        wavelet_vhighfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_third_peak]]
                    elif (freq_of_third_peak < freq_of_strongest_peak) and (freq_of_third_peak < freq_of_secondary_peak):
                       #if the third freq is the lowest, move the other two up
                        wavelet_vhighfreq_band = np.copy(wavelet_highfreq_band)
                        wavelet_highfreq_band = np.copy(wavelet_lowfreq_band)
                        wavelet_lowfreq_band[0:col] = pd.Series(np.repeat(np.nan, col))
                        wavelet_lowfreq_band[col] = freq_of_third_peak
                        
                        wavelet_vhighfreq_band_height = np.copy(wavelet_highfreq_band_height)
                        wavelet_highfreq_band_height = np.copy(wavelet_lowfreq_band_height)
                        wavelet_lowfreq_band_height[0:col] = pd.Series(np.repeat(np.nan, col))
                        wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_third_peak]]
                        
                        prev_vhighfreq = np.copy(prev_highfreq)
                        prev_highfreq = np.copy(prev_lowfreq)
                        prev_lowfreq = np.nan
                        print('INSERT THIRD FREQ AS LOW, SHIFT UP OTHER TWO FREQ')
                    else:
                        #if the third freq is in the middle: set vhigh to old high, and the high is the newfound third freq
                        wavelet_vhighfreq_band = np.copy(wavelet_highfreq_band)
                        wavelet_highfreq_band[0:col] = pd.Series(np.repeat(np.nan, col))
                        wavelet_highfreq_band[col] = freq_of_third_peak
                        
                        wavelet_vhighfreq_band_height = np.copy(wavelet_highfreq_band_height)
                        wavelet_highfreq_band_height[0:col] = pd.Series(np.repeat(np.nan, col))
                        wavelet_highfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_third_peak]]
                        
                        prev_vhighfreq = np.copy(prev_highfreq)
                        prev_highfreq = np.nan
                        print('INSERT THIRD FREQ IN MIDDLE, SHIFT UP HIGH TO VERY HIGH')


                elif (freq_of_strongest_peak + wavelet_band_width > freq_of_third_peak) and (freq_of_strongest_peak - wavelet_band_width < freq_of_third_peak):
                    print('Cannot distinguish well between strong and third frequencies, sample: ' + str(col))
                elif (freq_of_secondary_peak + wavelet_band_width > freq_of_third_peak) and (freq_of_secondary_peak - wavelet_band_width < freq_of_third_peak):
                    print('Cannot distinguish well between secondary and third frequencies, sample: ' + str(col))
                elif (prev_vhighfreq + wavelet_band_width > freq_of_third_peak) and (prev_vhighfreq - wavelet_band_width < freq_of_third_peak):
                    wavelet_vhighfreq_band[col] = freq_of_third_peak
                    wavelet_vhighfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_third_peak]]
                elif (prev_lowfreq + wavelet_band_width > freq_of_third_peak) and (prev_lowfreq - wavelet_band_width < freq_of_third_peak):
                    wavelet_lowfreq_band[col] = freq_of_third_peak
                    wavelet_lowfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_third_peak]]
                elif (prev_highfreq + wavelet_band_width > freq_of_third_peak) and (prev_highfreq - wavelet_band_width < freq_of_third_peak):
                    wavelet_highfreq_band[col] = freq_of_third_peak
                    wavelet_highfreq_band_height[col] = cwtmatr_norm_height[:,col][peak_indices[pos_of_third_peak]]
                else:
                    #print('Third freq cannot be classified, sample: ' + str(col) + '. It is ' + str(freq_of_third_peak) + ' - whereas prev low is ' + str(prev_lowfreq) + ' and prev high is ' + str(prev_highfreq) + ' and prev very high freq is ' + str(prev_vhighfreq))
                    tmp=0
            except IndexError as error:
            #only set previous value to nan if there have not yet been any values in this frequency band
                if col==0 or np.isnan(prev_vhighfreq):
                    prev_vhighfreq = np.nan

        # for the next column - set new frequencies that you want to assign each band to
        if col == 0 :
            prev_lowfreq = wavelet_lowfreq_band[col] 
            prev_highfreq = wavelet_highfreq_band[col] 
            prev_vhighfreq = wavelet_vhighfreq_band[col] 
        else:
            #the updated prev_freq is based on an average of the last N points
            if (col>0) and (col<=num_to_avg):
                start=0
            else:
                start=col-num_to_avg
            mean_lowfreq = np.nanmean(wavelet_lowfreq_band[start:col])
            mean_highfreq = np.nanmean(wavelet_highfreq_band[start:col])
            mean_vhighfreq = np.nanmean(wavelet_vhighfreq_band[start:col])
            #only update prev freq with a value that is not nan - don't introduce nans
            if (np.isnan(mean_lowfreq) == False) and (mean_highfreq - wavelet_band_width > mean_lowfreq):
                prev_lowfreq = mean_lowfreq
            if (np.isnan(mean_highfreq) == False) and (mean_lowfreq + wavelet_band_width < mean_highfreq):
                prev_highfreq = mean_highfreq
            if np.isnan(mean_vhighfreq) == False and (mean_highfreq + wavelet_band_width < mean_vhighfreq):
                prev_vhighfreq = mean_vhighfreq
                
        '''if (col%(100*450)) == 0:
            print(col)
            plt.figure()
            plt.plot(freq[0:50], cwtmatr_norm_height[:,col], '.')
            plt.plot(freq[0:50][peak_indices], cwtmatr_norm_height[:,col][peak_indices], '*')
            plt.show()
            #will see two peaks but may print nan if the freq cannot be classified
            print('highfreq ' + str(wavelet_highfreq_band[col]))
            print('lowfreq ' + str(wavelet_lowfreq_band[col]))
            print('veryhighfreq ' + str(wavelet_vhighfreq_band[col]))
            print('prev lowfreq' + str(prev_lowfreq))
            print('prev_highfreq' + str(prev_highfreq))
            print('prev_vhighfreq' + str(prev_vhighfreq))'''
        

    #censor the periodicity array and peak num arrays
    periodicity_percent_spectrum_above_halfmax_censored = np.copy(periodicity_percent_spectrum_above_halfmax)
    periodicity_percent_spectrum_above_halfmax_censored[censoring_arr_full] = np.nan
    wavelet_peak_num_censored = np.copy(wavelet_peak_num)
    wavelet_peak_num_censored[censoring_arr_full] = np.nan
    wavelet_vhighfreq_band[censoring_arr_full] = np.nan
    wavelet_highfreq_band[censoring_arr_full] = np.nan
    wavelet_lowfreq_band[censoring_arr_full] = np.nan
    wavelet_vhighfreq_band_height[censoring_arr_full] = np.nan
    wavelet_highfreq_band_height[censoring_arr_full] = np.nan
    wavelet_lowfreq_band_height[censoring_arr_full] = np.nan

     ################################ PLOT WAVELET TRANSFORM ########################################
    fig = plt.figure(figsize = (15,5))
    plt.pcolormesh(time_array, freq[0:50], cwtmatr_norm_height, cmap='jet', shading = 'auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (breaths/s)')
    plt.colorbar()
    plt.savefig(output_name +  '_wavelet_transform.png')
    plt.close()

    del peak_properties
    del cwtmatr
    del cwtmatr_norm_height
    del indices_above_halfmax
    gc.collect()
    
    return periodicity_percent_spectrum_above_halfmax, periodicity_percent_spectrum_above_halfmax_censored, wavelet_peak_num, wavelet_peak_num_censored, wavelet_vhighfreq_band*60, wavelet_highfreq_band*60, wavelet_lowfreq_band*60, wavelet_vhighfreq_band_height, wavelet_highfreq_band_height, wavelet_lowfreq_band_height

def get_entropy(trace_smoothed, entropy_type, m_val, censoring_arr_full):
    '''Compute the predictability of the time series in a window by calculating sample entropy'''
    #censor the necessary samples before computing entropy
    trace_smoothed_censored = np.copy(trace_smoothed)
    trace_smoothed_censored[censoring_arr_full] = np.nan
    
    #reshape from (n,) to (n,1)
    signal = np.array(trace_smoothed_censored).reshape(-1,1)
    
    #compute entropy: m is the size of a template for which the remaining data is scanned to see if this template repeats 
    # within a tolerance of r. A and B are the number of template matches for template sizes of m and m+1, respectively. The sample
    # entropy output contains values for all m values from 0 to the specified value.
    try:
        if entropy_type == 'Sample':
            entropy_allm, A, B = EH.SampEn(signal, m=m_val, tau = 1, r = 0.1*np.nanstd(signal), Logx = np.exp(1))
        elif entropy_type == 'Approximate':
            entropy_allm, phi = EH.ApEn(signal, m=m_val, tau = 1, r = 0.1*np.nanstd(signal), Logx = np.exp(1))
        else:
            print('Set entropy type as sample or approximate')

        #extract the last entropy value (corresponds to the desired m value)
        desired_entropy = entropy_allm[-1]
        
    except AssertionError as error:
        #in the case where the entire provided signal is NaNs (eg entire window is censored) - will get an error when computing entropy
        desired_entropy = np.nan
        
    return desired_entropy

def get_entropy_in_inst_window(trace_smoothed, censoring_arr_full, window_size_seconds, sampling_rate, tot_num_samples):
    '''this function repeats the get_entropy() function in a for loop of a few seconds - extract instantaneous metric of entropy in this 
    short window'''
    #define window size
    short_window_start_sample = 0
    short_window_width = window_size_seconds*sampling_rate
    
    #create empty array to store results
    entropy_inst = np.zeros(int(tot_num_samples/short_window_width))
    
    #iterate over multiple windows
    count = 0
    while short_window_start_sample + short_window_width <= tot_num_samples:
        
        #define window end
        short_window_end_sample = short_window_start_sample + short_window_width
        
        #extract entropy in that window
        entropy_inst[count]= get_entropy(trace_smoothed[short_window_start_sample:short_window_end_sample],'Sample',6,
                                        censoring_arr_full[short_window_start_sample:short_window_end_sample])
        
        #set next window
        short_window_start_sample = short_window_end_sample
        count = count+1
    #since the results will be 1 value per window - repeat that value to get an array of the same length as original data
    entropy_inst_full_length = np.repeat(entropy_inst, short_window_width)
    
    return entropy_inst, entropy_inst_full_length

def downsample_to_once_per_sec(series_to_downsample, tot_num_samples, tot_length_seconds):
    sampling_rate = int(tot_num_samples/tot_length_seconds)
    series_reshaped = series_to_downsample.to_numpy().reshape((sampling_rate, tot_length_seconds), order = 'F')
    
    #if there are only NaNs in the entire second, will cause runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        series_downsampled = np.nanmean(series_reshaped, axis=0)
    return pd.Series(series_downsampled)
    

def extract_all_pulseox_metrics(raw_trace_arr, df_censoring, large_window_width, large_window_overlap, window_length, tot_num_samples,
                             tot_length_seconds, output_name, h, d, pr, pl_max, width_val, wavelet_band_width, wavelet_peak_dist,
                               num_bands_to_detect, invert_bool, num_to_avg):
    ''' This function takes an input pulseox trace (assumed to be multiple minutes long with a 
    sampling rate of 450 samples/s) and computes the instantaneous respiration rate. The window_length argument refers
    the window within which the instantaneous window is computed (in seconds).'''
    sampling_rate = int(tot_num_samples/tot_length_seconds)
    time_array = np.arange(start=0, stop=tot_length_seconds , step=1/sampling_rate)
    
    #the censoring df is generated from the EPI, so it has length tot_length_seconds 
    # multiply it by sampling rate to get array of length tot_num_samples - convert so True represents points that ARE censored
    censoring_arr_full = np.repeat(np.array(df_censoring), sampling_rate) == False
    indices_of_censored_samples = np.where(censoring_arr_full ==1)[0]
    
    ######################################### PREPROCESSING #####################################
    #denoise
    trace_smoothed = denoise(raw_trace_arr, sampling_rate, invert_bool)
    del raw_trace_arr
    gc.collect()
    
    #extract the heart beat indices
    beat_indices, beats_bool, beats_toplot, beat_prominence = find_heart_beats(trace_smoothed, h, d, pr, pl_max, width_val)
    #find the location of censored breaths within the breath_indices_window
    location_of_censored = np.where(np.isin(beat_indices,indices_of_censored_samples))
    
    ######################################### EXTRACT INSTANTANEOUS METRICS ######################
    #resp rate in rolling window - per sample
    HR_inst, HR_inst_censored = get_HR_inst(beats_bool, censoring_arr_full, window_length, sampling_rate)
    
    #extract period between breaths - per beat pair
    period_btw_beats, period_ssd = get_period(beats_bool, sampling_rate)
    
    #extract HRV - std/rmssd of period in rolling window of 4 beats - per beat
    hrv_inst_std_period, hrv_inst_rmssd_period = get_inst_hrv(period_btw_beats, period_ssd, 4)
    
    #extract PVI - % var in amplitudes in rolling window of 9 beats (use min 9 to be sure to cover min 1 complete resp cycle 
    #- in case resp is slow (~60) and HR fast (~500)- per beat
    pvi_inst = get_PVI(beat_prominence, 9)
    
    #get the width of the pulse (at 1/4, 1/2 heights and base) - per beat
    width_quart, width_half, width_base= get_pulse_shape_metrics(trace_smoothed, beat_indices,sampling_rate)
    
     #extract periodicty - % wavelet above halfmax - per sample
    periodicty_percent_above_halfmax, periodicty_percent_above_halfmax_cens, wavelet_peak_num, wavelet_peak_num_censored, HR_from_wavelet_vhf_censored, HR_from_wavelet_hf_censored, HR_from_wavelet_lf_censored, wavelet_peak_height_vhf, wavelet_peak_height_hf, wavelet_peak_height_lf = get_periodiocity_wavelet(trace_smoothed,censoring_arr_full,sampling_rate,time_array, tot_num_samples,output_name,wavelet_band_width,wavelet_peak_dist, num_bands_to_detect, int(num_to_avg))
    
    #extract entropy in a 5s window 
    with warnings.catch_warnings():
        warnings.simplefilter(action = "ignore", category = RuntimeWarning)
        entropy_inst, entropy_inst_full_length = get_entropy_in_inst_window(trace_smoothed, censoring_arr_full, 5,
                                                                            sampling_rate, tot_num_samples)
    
    ########################################## PLOT INSTANTANEOUS METRICS - for QC ###############
    #for the metrics where there is only one value per breath, repeat same value until next breath
    period_btw_beats_toplot = repeat_values_for_plotting(period_btw_beats, beats_bool, beat_indices)
    hrv_inst_std_period_toplot = repeat_values_for_plotting(hrv_inst_std_period, beats_bool, beat_indices)
    hrv_inst_rmssd_period_toplot = repeat_values_for_plotting(hrv_inst_rmssd_period, beats_bool, beat_indices)
    pvi_inst_toplot = repeat_values_for_plotting(pvi_inst, beats_bool, beat_indices)
    width_quart_toplot = repeat_values_for_plotting(width_quart, beats_bool, beat_indices)
    
    #plot each 20s segment
    samples_per_iteration = int(sampling_rate*20)
    start = 0
    end = samples_per_iteration
    while end < tot_num_samples: 
        fig, ax = plt.subplots(figsize = (15,5))
        #plot the respiration trace and the detected breaths to make sure that they were properly detected
        ax.plot(time_array[start:end], 2*trace_smoothed[start:end]+ np.nanmax(HR_inst), label = 'Smoothed Pulseox Trace')
        ax.plot(time_array[start:end], 2*beats_toplot[start:end]+ np.nanmax(HR_inst), '.', label = 'Detected Beat')
        ax.plot(time_array[start:end], HR_inst[start:end], label = 'Resp Rate')
        ax.plot(time_array[start:end], 50*period_btw_beats_toplot[start:end], label = 'Period (x50)')
        ax.plot(time_array[start:end], 1000*hrv_inst_std_period_toplot[start:end], label = 'HRV-std (x1000)')
        ax.plot(time_array[start:end], 1000*hrv_inst_rmssd_period_toplot[start:end], label = 'HRV-rmssd (x1000)')
        ax.plot(time_array[start:end], pvi_inst_toplot[start:end], label = 'PVI')
        ax.plot(time_array[start:end], 100*width_quart_toplot[start:end], label = 'Width at quarter height (x100)')
        ax.plot(time_array[start:end], 5*periodicty_percent_above_halfmax[start:end], label = 'Periodicity (% wavelet above HM)(x5)')
        ax.plot(time_array[start:end], 50*wavelet_peak_num[start:end], label = 'Number of wavelet peaks (x50)')
        ax.plot(time_array[start:end], 1000*entropy_inst_full_length[start:end], label = 'Entropy (x1000)')
        ax.fill_between(time_array[start:end], 0, 1, where=censoring_arr_full[start:end], facecolor='red', alpha=0.2,
                        transform=ax.get_xaxis_transform())
        ax.set_xlabel('Time (s)')
        ax.set_title('Quality Control Heart Beat Extraction')
        ax.legend()
        fig.savefig(output_name + '_start_' + str(int(time_array[start])) + 's.png')
        plt.close()
        start = start + samples_per_iteration
        end = end + samples_per_iteration
    
        ######################################### EXTRACT AVERAGE METRICS IN WINDOW ##################
    #create arrays to store the values for for all windows
    num_windows = 1+int((tot_length_seconds - large_window_width)/large_window_overlap)#numerator gives last start, frac gives num starts
    metrics_in_window = np.zeros((num_windows,21))
    if num_bands_to_detect==3:
        HR_vhigh_in_window = np.zeros((num_windows,2))
    
    #extract a time window
    large_window_start_realtime = 0 
    count = 0
    while large_window_start_realtime + large_window_width <= tot_length_seconds:
        
        #calculate when the time window should end (according to the length of the original, uncensored data)
        large_window_end_realtime = large_window_start_realtime + large_window_width
        large_window_start_samplenum = large_window_start_realtime*sampling_rate
        large_window_end_samplenum = large_window_end_realtime*sampling_rate
        metrics_in_window[count,0] = large_window_start_realtime
        metrics_in_window[count,1] = large_window_end_realtime
        
        #extract only the trace and beats within that window
        beat_indices_window_nocensor = (beat_indices >= large_window_start_samplenum) & (beat_indices < large_window_end_samplenum)
        #ignore the breaths that are in a censored area
        beat_indices_window_censor = np.copy(beat_indices_window_nocensor)
        beat_indices_window_censor[location_of_censored] = False
        
        #extract mean/std of instantaneous HR in that window
        metrics_in_window[count,2] = np.nanmean(HR_inst_censored[large_window_start_samplenum:large_window_end_samplenum])
        metrics_in_window[count,3] = np.nanstd(HR_inst_censored[large_window_start_samplenum:large_window_end_samplenum])
        
        #extract mean of instantaneous HRV in that window
        metrics_in_window[count,4] = np.mean(hrv_inst_std_period[beat_indices_window_censor])
        metrics_in_window[count,5] = np.mean(hrv_inst_rmssd_period[beat_indices_window_censor])
        
        #extract mean of instantaneous pvi in that window
        metrics_in_window[count,6] = np.mean(pvi_inst[beat_indices_window_censor])
        
        #extract mean of instantaneous periodicity in that window
        metrics_in_window[count,7] = np.nanmean(periodicty_percent_above_halfmax_cens[large_window_start_samplenum:large_window_end_samplenum])
        metrics_in_window[count,8] = np.nanmean(wavelet_peak_num_censored[large_window_start_samplenum:large_window_end_samplenum])
        metrics_in_window[count,9] = np.nanmean(HR_from_wavelet_hf_censored[large_window_start_samplenum:large_window_end_samplenum])
        metrics_in_window[count,10] = np.nanmean(HR_from_wavelet_lf_censored[large_window_start_samplenum:large_window_end_samplenum])
        metrics_in_window[count,11] = np.nanmean(wavelet_peak_height_hf[large_window_start_samplenum:large_window_end_samplenum])
        metrics_in_window[count,12] = np.nanmean(wavelet_peak_height_lf[large_window_start_samplenum:large_window_end_samplenum])
        
        #extract mean of instantaneous entropy in that window
        metrics_in_window[count,13] = np.mean(entropy_inst_full_length[large_window_start_samplenum:large_window_end_samplenum])
        
        #extract overall resp rate across whole window (can't censor breaths or will give inaccurate rate
        metrics_in_window[count,14] = (60/large_window_width)*(beat_indices[beat_indices_window_nocensor].size)
        
        #extract mean period and variability in period (RRV) across whole window
        metrics_in_window[count,15] = np.mean(period_btw_beats[beat_indices_window_censor])
        metrics_in_window[count,16] = np.std(period_btw_beats[beat_indices_window_censor])
        metrics_in_window[count,17] = np.mean(period_ssd[beat_indices_window_censor])**(1/2)
        
        #extract mean widths in that window
        metrics_in_window[count,18] = np.mean(width_quart[beat_indices_window_censor])
        metrics_in_window[count,19] = np.mean(width_half[beat_indices_window_censor])
        metrics_in_window[count,20] = np.mean(width_base[beat_indices_window_censor])
        
        #NOTE: I am NOT extracting the entropy across the whole window due to prohibitive memory/time constraints - also preliminary         
        #analysis showed that entropy across the whole window is very highly correlated with mean of instantaneous entropy.
        
        #only if I expect 3 wavelet freq, calc the means in windows
        if num_bands_to_detect==3:
            HR_vhigh_in_window[count, 0] = np.nanmean(HR_from_wavelet_vhf_censored[large_window_start_samplenum:large_window_end_samplenum])
            HR_vhigh_in_window[count, 1] = np.nanmean(wavelet_peak_height_vhf[large_window_start_samplenum:large_window_end_samplenum])
        
        #set the start time of the next window in realtime
        large_window_start_realtime = large_window_end_realtime - large_window_overlap
        count = count+1
    
    ######################################## SAVE OUTPUTS ######################################
    df_onesample_per_window = pd.DataFrame(metrics_in_window, columns = ['Window start time','Window end time','Instantaneous HR-window mean', 'Instantaneous HR - window std', 'Instantaneous HRV period std-window mean','Instantanous HRV period rmssd-window mean', 'Instantaneous PVI-window mean','Instantaneous periodicity-window mean', 'Instantaneous number of wavelet peaks-window mean','Instantaneous highfreq HR from wavelet-window mean', 'Instantaneous lowfreq HR from wavelet-window mean', 'Instantaneous highfreq peak height-window mean','Instantaneous lowfreq peak height-window mean','Instantaneous entropy-window mean','HR-overall window','Period-overall window mean', 'HRV-overall period window std', 'HRV-overall period window rmssd', 'Width at quarter height-overall window mean', 'width at half height-overall window mean','Width at base-overall window mean'])
    if num_bands_to_detect==3:
        df_onesample_per_window.insert(9, 'Instantaneous veryhighfreq HR from wavelet-window mean', HR_vhigh_in_window[:, 0])
        df_onesample_per_window.insert(12, 'Instantaneous veryhighfreq peak height-window mean', HR_vhigh_in_window[:, 1])
    
    df_onesample_per_window.to_csv(output_name + "_per_window.csv")
    
    ###################################### Also return HR per sec - for comparison with SAII results #####################
    HR_inst_per_sec = downsample_to_once_per_sec(HR_inst, tot_num_samples, tot_length_seconds)
    HR_inst_cens_per_sec = downsample_to_once_per_sec(pd.Series(HR_inst_censored), tot_num_samples, tot_length_seconds)
    df_HR_onesample_per_sec = pd.concat([HR_inst_per_sec, HR_inst_cens_per_sec], axis = 1, ignore_index = True)
    df_HR_onesample_per_sec.columns = ['HR_inst','HR_inst_censored']
    df_HR_onesample_per_sec.to_csv(output_name + "_HR_per_sec.csv")
            
    #delete unnecessary ones that are per sample
    del time_array
    del trace_smoothed
    del location_of_censored
    del censoring_arr_full
    del indices_of_censored_samples
    del beat_indices
    del beats_bool
    del beats_toplot
    del beat_prominence
    del HR_inst
    del HR_inst_censored
    del periodicty_percent_above_halfmax
    del periodicty_percent_above_halfmax_cens
    del wavelet_peak_num
    del wavelet_peak_num_censored
    del HR_from_wavelet_vhf_censored
    del HR_from_wavelet_hf_censored
    del HR_from_wavelet_lf_censored
    del wavelet_peak_height_lf
    del wavelet_peak_height_hf 
    del wavelet_peak_height_vhf 
    gc.collect()
    

