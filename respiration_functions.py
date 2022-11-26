#!/usr/bin/env python
# coding: utf-8

""" Purpose: extract metrics such as respiration rate/period/periodicity from the raw respiration trace.
COMPUTE INSTANTANEOUS METRICS:
1. rate (breaths/time in short rolling window, per sample res, avg across samples in long window)
2. period (diff between breaths, per breath res, avg across breaths in long window)

COMPUTE INSTANTANEOUS VARABILITY:
3. RRV (std of period of breaths in short window - will indicate transitions in rate, per breath res, avg across samples in long window)
4. RRV (RMSSD between breaths in short window - indicates transitions in rate, per breath res, avg across samples in long window)
5. periodicity (% wavelet above halfmax - indicates inst variability, per sample res, avg across samples in long window)
6. predictability (entropy in short window - indicates transitions, per sample res, avg across samples in long window)

COMPUTE AVERAGE METRICS ACROSS ENTIRE LONG WINDOW:
7. mean rate (breaths/time across long window)

COMPUTE VARIABILITY METRICS ACROSS ENTIRE WINDOW:
8. std of inst rate across the long window
9. RRV (std of period across all breaths in long window - indicates variability in window)
10. RRV (RMSSD between all breaths in long window - indicates variability in window)
11. predictability (entropy in long window - indicates predictability in whole window) 
 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.signal import find_peaks
import warnings
import EntropyHub as EH
import warnings

def repeat_values_for_plotting(data_to_repeat, breaths_bool, breath_indices):
    '''For metrics where there is only 1 value per breath - duplicate the value until the next breath to create array
    with the length of all samples in original data - that way it can be plotted on the same graph as original trace.'''
    
    full_length_array = pd.Series(np.repeat(np.nan, len(breaths_bool)))
    
    for i in range(0, len(breath_indices)-1):
        full_length_array[breath_indices[i]:breath_indices[i+1]] = np.repeat(data_to_repeat[i], breath_indices[i+1]-breath_indices[i])
        
    return full_length_array        
        
def denoise_detrend(raw_trace, sampling_rate, invert_bool):
    '''function to smooth and detrend the data'''
    #convert the input array to a pandas Series, since the rolling window function is a pandas function
    trace = pd.Series(raw_trace)
    
    #smooth the trace by taking the mean within a rolling window of 80 samples (Gaussian weighted mean, std =20)
    #using the Gaussian weighting makes every element in the smoothed series unique with higher precision
    trace_smoothed = raw_trace.rolling(40, center = True, min_periods = 1, win_type = 'gaussian').mean(std=10)
    
    #detrend the smoothed raw trace by subtracting the mean across 1 s (if necessary, invert the trace along y-axis first
    #inversion is to account for the fact that sometimes the resp pillow is placed backwards, so inspiration = down
    trend = trace_smoothed.rolling(sampling_rate, center = True, min_periods = 1).mean()
    if invert_bool:
        trace_smoothed_detrend_init = -1*trace_smoothed + trend
    else:
        trace_smoothed_detrend_init = trace_smoothed - trend
    trace_smoothed_detrend = trace_smoothed_detrend_init.reset_index(drop= True)
    
    return trace_smoothed_detrend

def find_breaths(resp_trace_smoothed_detrend, h, d, pr, t, pl_min, pl_max, wl):
    '''function to detect breaths throughout entire trace using peak detection algorithm'''
    #find most prominent peaks
    breath_indices, _ = find_peaks(resp_trace_smoothed_detrend, height = h, distance = d, prominence = pr, threshold = t,
                                   plateau_size = (pl_min,pl_max), wlen = wl)
    
    #create boolean breath array
    breaths_bool = pd.Series(np.repeat(0, len(resp_trace_smoothed_detrend)))
    breaths_bool[breath_indices] = 1

    #create array for plotting each breath on top of the original trace
    breaths_toplot = (breaths_bool*resp_trace_smoothed_detrend).replace(0,np.nan)
        
    return breath_indices, breaths_bool, breaths_toplot

def get_resp_rate_inst(breaths, censoring_arr_full, window_length, sampling_rate):
    '''function to calculate instantaneous respiration rate (based on # of breaths in a rolling time window of a few sec)'''
    
    #calculate the number of breaths in a time window of the specified length
    breath_sum = breaths.rolling(window_length*sampling_rate, center = True).sum()

    #convert the sum of breaths to respiration rate in breaths/min
    resp_rate = (60/window_length)*breath_sum

    #smooth the respiration rate trace such that there are fewer bumps when the rolling window encounters a new breath
    resp_rate_smooth = resp_rate.rolling(2*sampling_rate, center = True, win_type = 'gaussian').mean(std=100)
    
    #censor the necessary samples by setting them to nan
    resp_rate_smooth_censored = np.copy(resp_rate_smooth)
    resp_rate_smooth_censored[censoring_arr_full] = np.nan
        
    return resp_rate_smooth, resp_rate_smooth_censored

def get_period(breath_indices, sampling_rate):
    
    #find the difference between breaths (ie between consecutive elements of the location array), divide by sampling_rate to get seconds
    period_btw_breaths = pd.Series(breath_indices).diff(periods=1)/sampling_rate
    
    #find successive differences between consecutive periods and square
    period_ssd = pd.Series(period_btw_breaths).diff(periods=1)**2
        
    return period_btw_breaths, period_ssd

def get_inst_rrv(period_btw_breaths, period_ssd, window_size):
    '''function to extract the respiratory rate variability (ie a metric of how much the resp rate changes over time) using the standard
    deviation of resp rate in a window (either short-inst or long) and root mean square of successive differences (RMSSD) between 
    consectutive breaths.
    3,4,9,10 - compute directly in custom rolling window of breaths (get_rrv for inst and avg)'''
    
    #compute std of period btw breaths in a window of n breaths
    rrv_std_period = period_btw_breaths.rolling(window_size, min_periods = 0, center = True).std()
    
    #compute RMSSD (root mean square of successive differences in period) by finding mean (then root) in a window of n breaths
    #rrv_rmssd_period = pd.Series(np.repeat(np.nan, len(period_btw_breaths)))
    rrv_rmssd_period = period_ssd.rolling(window_size-1, min_periods = 0, center = True).mean()**(1/2)
    
    return rrv_std_period, rrv_rmssd_period

def get_periodiocity_wavelet(resp_trace_smoothed_detrend, censoring_arr_full, sampling_rate, time_array,tot_num_samples, output_name):
    '''this function computes the wavelet transform (frequency spectrum at each point in time - very robust to noise) then 
    examines how power is concentrated across the frequencies'''
 
    # pad the trace to reduce edge effects
    pad_samples = 500
    padded_trace = np.pad(resp_trace_smoothed_detrend, pad_width = pad_samples, mode = 'reflect')
    
    #find wavelet transform with a morlet wavelet of width 8
    freq = np.linspace(1, sampling_rate/2, 2000)
    widths = (8.0)*sampling_rate / (2*freq*np.pi)
    cwtmatr = np.abs(signal.cwt(padded_trace, signal.morlet2, widths, w = 8.)[0:100,pad_samples:tot_num_samples+pad_samples])
    
    #normalize columns to max height to 1
    cwtmatr_norm_height = cwtmatr/np.max(cwtmatr) 

    #calculate the % that are located above the half max (gives approx measure of spread)
    periodicity_percent_spectrum_above_halfmax = pd.Series(np.repeat(np.nan, tot_num_samples))
    halfmax = np.max(cwtmatr, axis = 0)/2
    for col in range(0,tot_num_samples):
        indices_above_halfmax = np.where(cwtmatr[:,col] >= halfmax[col])[0]
        periodicity_percent_spectrum_above_halfmax[col] = 100*len(indices_above_halfmax)/100 #100 points per spectrum

    #censor the periodicity array
    periodicity_percent_spectrum_above_halfmax_censored = np.copy(periodicity_percent_spectrum_above_halfmax)
    periodicity_percent_spectrum_above_halfmax_censored[censoring_arr_full] = np.nan
    
     ################################ PLOT WAVELET TRANSFORM ########################################
    fig = plt.figure(figsize = (15,5))
    plt.pcolormesh(time_array, freq[0:100], cwtmatr_norm_height, cmap='jet', shading = 'auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (breaths/s)')
    plt.colorbar()
    plt.savefig(output_name +  '_wavelet_transform.png')
    
    return periodicity_percent_spectrum_above_halfmax, periodicity_percent_spectrum_above_halfmax_censored

def get_entropy(resp_trace_smoothed_detrend, entropy_type, m_val, censoring_arr_full):
    '''Compute the predictability of the time series in a window by calculating sample entropy. m is the size of a template for which the 
    remaining data is scanned to see if this template repeats within a tolerance of r. A and B are the number of template matches for 
    template sizes of m and m+1, respectively. The sample entropy output contains values for all m values from 0 to the specified value'''
    
    #censor the necessary samples before computing entropy
    resp_trace_smoothed_detrend_censored = np.copy(resp_trace_smoothed_detrend)
    resp_trace_smoothed_detrend_censored[censoring_arr_full] = np.nan
    
    #reshape from (n,) to (n,1)
    signal = np.array(resp_trace_smoothed_detrend_censored).reshape(-1,1)
    
    #compute entropy
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

def downsample_to_once_per_sec(series_to_downsample, tot_num_samples, tot_length_seconds):
    sampling_rate = int(tot_num_samples/tot_length_seconds)
    series_reshaped = series_to_downsample.to_numpy().reshape((sampling_rate, tot_length_seconds), order = 'F')
    
    #if there are only NaNs in the entire second, will cause runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        series_downsampled = np.nanmean(series_reshaped, axis=0)
    return pd.Series(series_downsampled)

def get_entropy_in_inst_window(resp_trace_smoothed_detrend, censoring_arr_full, window_size_seconds, sampling_rate, tot_num_samples):
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
        entropy_inst[count]= get_entropy(resp_trace_smoothed_detrend[short_window_start_sample:short_window_end_sample],'Sample',4,
                                        censoring_arr_full[short_window_start_sample:short_window_end_sample])
        
        #set next window
        short_window_start_sample = short_window_end_sample
        count = count+1
    #since the results will be 1 value per window - repeat that value to get an array of the same length as original data
    entropy_inst_full_length = np.repeat(entropy_inst, short_window_width)
    
    return entropy_inst, entropy_inst_full_length


def extract_all_resp_metrics(raw_resp_trace_arr, df_censoring, large_window_width, large_window_overlap, window_length, tot_num_samples,
                             tot_length_seconds, output_name, invert_bool, h, d, pr, t, pl_min, pl_max, wl):
    ''' This function takes an input respiration trace (assumed to be multiple minutes long with a 
    sampling rate of 225 samples/s) and computes the instantaneous respiration rate. The window_length argument refers
    the window within which the instantaneous window is computed (in seconds).'''
    sampling_rate = int(tot_num_samples/tot_length_seconds)
    time_array = np.arange(start=0, stop=tot_length_seconds , step=1/sampling_rate)
    
    #the censoring df is generated from the EPI, so it has length tot_length_seconds 
    # multiply it by sampling rate to get array of length tot_num_samples - convert so True represents points that ARE censored
    censoring_arr_full = np.repeat(np.array(df_censoring), sampling_rate) == False
    indices_of_censored_samples = np.where(censoring_arr_full ==1)[0]
    
    ######################################### PREPROCESSING #####################################
    #denoise and detrend
    resp_trace_smoothed_detrend = denoise_detrend(raw_resp_trace_arr, sampling_rate, invert_bool)
    #extract the breath indices
    breath_indices, breaths_bool, breaths_toplot = find_breaths(resp_trace_smoothed_detrend, h, d, pr, t, pl_min, pl_max,wl)
    #find the location of censored breaths within the breath_indices_window
    location_of_censored = np.where(np.isin(breath_indices,indices_of_censored_samples))
    
    ######################################### EXTRACT INSTANTANEOUS METRICS ######################
    #resp rate in rolling window - per sample
    resp_rate_inst, resp_rate_inst_censored = get_resp_rate_inst(breaths_bool, censoring_arr_full, window_length, sampling_rate)
    
    #extract period between breaths - per breath pair
    period_btw_breaths, period_ssd = get_period(breath_indices, sampling_rate)
    
    #extract RRV - std/rmssd of period in rolling window of 4 breaths - per breath
    rrv_inst_std_period, rrv_inst_rmssd_period = get_inst_rrv(period_btw_breaths, period_ssd, 4)
    
    #extract periodicty - % wavelet above halfmax - per sample
    periodicty_percent_above_halfmax, periodicty_percent_above_halfmax_cens = get_periodiocity_wavelet(resp_trace_smoothed_detrend,
                                                                                                       censoring_arr_full,sampling_rate,
                                                                                                       time_array, tot_num_samples,
                                                                                                       output_name)
    
    #extract entropy in a 5s window 
    with warnings.catch_warnings():
        warnings.simplefilter(action = "ignore", category = RuntimeWarning)
        entropy_inst, entropy_inst_full_length = get_entropy_in_inst_window(resp_trace_smoothed_detrend, censoring_arr_full, 5,
                                                                            sampling_rate,tot_num_samples)
    
     ########################################## PLOT INSTANTANEOUS METRICS - for QC ###############
    #for the metrics where there is only one value per breath, repeat same value until next breath
    period_btw_breaths_toplot = repeat_values_for_plotting(period_btw_breaths, breaths_bool, breath_indices)
    rrv_inst_std_period_toplot = repeat_values_for_plotting(rrv_inst_std_period, breaths_bool, breath_indices)
    rrv_inst_rmssd_period_toplot = repeat_values_for_plotting(rrv_inst_rmssd_period, breaths_bool, breath_indices)
    
    #plot each 30s segment
    samples_per_iteration = int(sampling_rate*30)
    start = 0
    end = samples_per_iteration
    while end < tot_num_samples: 
        fig, ax = plt.subplots(figsize = (15,5))
        #plot the respiration trace and the detected breaths to make sure that they were properly detected
        ax.plot(time_array[start:end], resp_trace_smoothed_detrend[start:end]+60, label = 'Smoothed Resp Trace')
        ax.plot(time_array[start:end], breaths_toplot[start:end]+60, '*', label = 'Detected Breath')
        ax.plot(time_array[start:end], resp_rate_inst[start:end], label = 'Resp Rate')
        ax.plot(time_array[start:end], 50*period_btw_breaths_toplot[start:end], label = 'Period (x50)')
        ax.plot(time_array[start:end], 100*rrv_inst_std_period_toplot[start:end], label = 'RRV-std (x100)')
        ax.plot(time_array[start:end], 100*rrv_inst_rmssd_period_toplot[start:end], label = 'RRV-rmssd (x100)')
        ax.plot(time_array[start:end], periodicty_percent_above_halfmax[start:end], label = 'Periodicity (% wavelet above HM)')
        ax.plot(time_array[start:end], 100*entropy_inst_full_length[start:end], label = 'Entropy (x100)')
        ax.fill_between(time_array[start:end], 0, 1, where=censoring_arr_full[start:end], facecolor='red', alpha=0.2,
                        transform=ax.get_xaxis_transform())
        ax.set_xlabel('Time (s)')
        ax.set_title('Quality Control Breath Extraction')
        ax.legend()
        fig.savefig(output_name + '_start_' + str(int(time_array[start])) + 's.png')
        plt.close()
        start = start + samples_per_iteration
        end = end + samples_per_iteration
    
    ######################################### EXTRACT AVERAGE METRICS IN WINDOW ##################
    #create arrays to store the values for for all windows
    num_windows = 1+int((tot_length_seconds - large_window_width)/large_window_overlap)#numerator gives last start, frac gives num starts
    metrics_in_window = np.zeros((num_windows,13))
    
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
        
        #extract only the trace and breaths within that window
        breath_indices_window_nocensor = (breath_indices >= large_window_start_samplenum) & (breath_indices < large_window_end_samplenum)
        #ignore the breaths that are in a censored area
        breath_indices_window_censor = np.copy(breath_indices_window_nocensor)
        breath_indices_window_censor[location_of_censored] = False
        
        #extract mean/std of instantaneous resp rate in that window
        metrics_in_window[count,2] = np.nanmean(resp_rate_inst_censored[large_window_start_samplenum:large_window_end_samplenum])
        metrics_in_window[count,3] = np.nanstd(resp_rate_inst_censored[large_window_start_samplenum:large_window_end_samplenum])
        
        #extract mean of instantaneous RRV in that window
        metrics_in_window[count,4] = np.mean(rrv_inst_std_period[breath_indices_window_censor])
        metrics_in_window[count,5] = np.mean(rrv_inst_rmssd_period[breath_indices_window_censor])
        
        #extract mean of instantaneous periodicity in that window
        metrics_in_window[count,6] = np.nanmean(periodicty_percent_above_halfmax_cens[large_window_start_samplenum:large_window_end_samplenum])
        
        #extract mean of instantaneous entropy in that window
        metrics_in_window[count,7] = np.mean(entropy_inst_full_length[large_window_start_samplenum:large_window_end_samplenum])
        
        #extract overall resp rate across whole window (can't censor breaths or will give inaccurate rate
        metrics_in_window[count,8] = (60/large_window_width)*(breath_indices[breath_indices_window_nocensor].size)
        
        #extract mean period and variability in period (RRV) across whole window
        metrics_in_window[count,9] = np.mean(period_btw_breaths[breath_indices_window_censor])
        metrics_in_window[count,10] = np.std(period_btw_breaths[breath_indices_window_censor])
        metrics_in_window[count,11] = np.mean(period_ssd[breath_indices_window_censor])**(1/2)
        
        #extract entropy across whole window (I have to do two half windows because of errors with memory allocation)
        large_window_end_samplenum_half = int(large_window_start_samplenum + sampling_rate*large_window_width/2)
        entropy_window_1half = get_entropy(resp_trace_smoothed_detrend[large_window_start_samplenum:large_window_end_samplenum_half],
                                          'Approximate', 4,
                                           censoring_arr_full[large_window_start_samplenum:large_window_end_samplenum_half])
        entropy_window_2half = get_entropy(resp_trace_smoothed_detrend[large_window_end_samplenum_half:large_window_end_samplenum],
                                          'Approximate', 4,
                                           censoring_arr_full[large_window_start_samplenum:large_window_end_samplenum_half])
        metrics_in_window[count,12] = np.mean([entropy_window_1half, entropy_window_2half])
        
        #set the start time of the next window in realtime
        large_window_start_realtime = large_window_end_realtime - large_window_overlap
        count = count+1
    
    ######################################## SAVE OUTPUTS ######################################
    df_onesample_per_window = pd.DataFrame(metrics_in_window, columns = ['Window start time','Window end time','Instantaneous resp rate-window mean', 'Instantaneous resp rate - window std', 'Instantaneous RRV period std-window mean','Instantanous RRV period rmssd-window mean', 'Instantaneous periodicity-window mean', 'Instantaneous entropy-window mean','Resp rate-overall window','Period-overall window mean', 'RRV-overall period window std', 'RRV-overall period window rmssd', 'Entropy-overall window'])
    
    df_onesample_per_window.to_csv(output_name + "_per_window.csv")
    
    ###################################### Also return RR per sec - for comparison with SAII results #####################
    RR_inst_per_sec = downsample_to_once_per_sec(resp_rate_inst, tot_num_samples, tot_length_seconds)
    RR_inst_cens_per_sec = downsample_to_once_per_sec(pd.Series(resp_rate_inst_censored), tot_num_samples, tot_length_seconds)
    df_RR_onesample_per_sec = pd.concat([RR_inst_per_sec, RR_inst_cens_per_sec], axis = 1, ignore_index = True)
    df_RR_onesample_per_sec.columns = ['RR_inst','RR_inst_censored']
    df_RR_onesample_per_sec.to_csv(output_name + "_RR_per_sec.csv")
