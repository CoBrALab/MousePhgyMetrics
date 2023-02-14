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
import ast
import os
import sys

######################################## Take the arguments from bash ##################################
analysis_type=sys.argv[1]
input_trace=os.path.abspath(sys.argv[2])
tot_length_seconds=ast.literal_eval(sys.argv[3])
output_name=os.path.abspath(sys.argv[4])
image_output_type=sys.argv[5]
peak_detection_parameter_csv=os.path.abspath(sys.argv[6])
invert_bool=ast.literal_eval(sys.argv[7])
window_length=ast.literal_eval(sys.argv[8])
fMRI_censoring_mask_csv=os.path.abspath(sys.argv[9])
fMRI_TR=ast.literal_eval(sys.argv[10])
large_window_width=ast.literal_eval(sys.argv[11])
large_window_overlap=ast.literal_eval(sys.argv[12])

if 'None' in peak_detection_parameter_csv:
    peak_detection_parameter_csv = None
if 'None' in fMRI_censoring_mask_csv:
    fMRI_censoring_mask_csv = None

######################################## FUNCTIONS #################################

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

def find_heart_beats(trace_smoothed, param_dict):
    
    #find most prominent peaks - setting a min width prevents detection of a bifurcated peak as two separate peaks
    if (param_dict['wlen'] is not None) & (param_dict['rel_height'] is not None):
        beat_indices, beat_properties = find_peaks(trace_smoothed, height = param_dict['height'], threshold = param_dict['threshold'], 
                                                    distance = param_dict['distance'], prominence = param_dict['prominence'],
                                                   width = param_dict['width'], wlen = param_dict['wlen'], 
                                                   rel_height = param_dict['rel_height'], plateau_size = param_dict['plateau_size'])
    elif (param_dict['wlen'] is None) & (param_dict['rel_height'] is None):
           beat_indices, beat_properties = find_peaks(trace_smoothed, height = param_dict['height'], 
                                                      threshold = param_dict['threshold'], distance = param_dict['distance'], 
                                                      prominence = param_dict['prominence'],width = param_dict['width'],  
                                                      plateau_size = param_dict['plateau_size'])
    elif (param_dict['wlen'] is not None) & (param_dict['rel_height'] is None):
           beat_indices, beat_properties = find_peaks(trace_smoothed, height = param_dict['height'], 
                                                      threshold = param_dict['threshold'], distance = param_dict['distance'], 
                                                      prominence = param_dict['prominence'],width = param_dict['width'],  
                                                      wlen = param_dict['wlen'], plateau_size = param_dict['plateau_size'])
    elif (param_dict['wlen'] is None) & (param_dict['rel_height'] is not None):
           beat_indices, beat_properties = find_peaks(trace_smoothed, height = param_dict['height'], 
                                                  threshold = param_dict['threshold'], distance = param_dict['distance'], 
                                                  prominence = param_dict['prominence'],width = param_dict['width'],  
                                                  rel_height = param_dict['rel_height'], plateau_size = param_dict['plateau_size'])
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
    
    if censoring_arr_full is not None:
        #censor the necessary samples by setting them to nan
        HR_smooth_censored = np.copy(HR_smooth)
        try:
            HR_smooth_censored[censoring_arr_full] = np.nan
        except:
            print("The duration of the plethysmography trace does not match the duration of the fMRI censoring csv. Check the number of samples in both csvs as well as the specified fMRI_TR and tot_length_seconds options.")

    else:
        HR_smooth_censored = None
        
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

def get_wavelet(trace_smoothed, sampling_rate, time_array, tot_num_samples, output_name, image_output_type):
    '''this function computes the wavelet transform (frequency spectrum at each point in time - very robust to noise) '''
 
    # pad the trace to reduce edge effects
    pad_samples = 500
    padded_trace = np.pad(trace_smoothed, pad_width = pad_samples, mode = 'reflect')
    
    #find wavelet transform with a morlet wavelet of width 8
    freq = np.linspace(1, sampling_rate/2, 1000)
    widths = (8.0)*sampling_rate / (2*freq*np.pi)
    cwtmatr = np.abs(signal.cwt(padded_trace, signal.morlet2, widths, w = 8.)[0:50,pad_samples:tot_num_samples+pad_samples])
    
    #normalize columns to max height to 1
    cwtmatr_norm_height = cwtmatr/np.max(cwtmatr) 

     ################################ PLOT WAVELET TRANSFORM ########################################
    fig = plt.figure(figsize = (12,4))
    plt.pcolormesh(time_array, 60*freq[0:50], cwtmatr_norm_height, cmap='jet', shading = 'auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (bpm)')
    plt.colorbar()
    if image_output_type == 'svg':
        plt.savefig(output_name +  'pleth_wavelet_transform.svg')
    else:
        plt.savefig(output_name +  'pleth_wavelet_transform.png')
    plt.close()

    return cwtmatr, cwtmatr_norm_height

def get_entropy(trace_smoothed, entropy_type, m_val, censoring_arr_full):
    '''Compute the predictability of the time series in a window by calculating sample entropy'''
    #censor the necessary samples before computing entropy
    trace_smoothed_censored = np.copy(trace_smoothed)
    if censoring_arr_full is not None:
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

                #if there is a censoring array, extract portion within the window
        if censoring_arr_full is not None:
            censoring_arr_window = censoring_arr_full[short_window_start_sample:short_window_end_sample]
        else:
            censoring_arr_window = None
        
        #extract entropy in that window
        entropy_inst[count]= get_entropy(trace_smoothed[short_window_start_sample:short_window_end_sample],'Sample',6,
                                        censoring_arr_window)
        
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

def process_user_csv(csv):
    ''' Check that the provided csv has the right columns. When the user provides a csv containing 'None' values or tuples, they are
    interpreted as a string by default. Convert them to actual Nonetype or tuple.'''
    df_strNone = pd.read_csv(csv, sep = '/', index_col = False)
    list_expected_cols = ['height', 'threshold', 'distance', 'prominence', 'width', 'wlen', 'rel_height', 'plateau_size']
    dict_trueNone = {}

    #check that each of the expected columns is present in the csv
    for expected_col in list_expected_cols:
        if expected_col not in df_strNone:
            raise Exception('The peak_detection_parameter_csv does not contain a column for ' + str(expected_col) + '. You must add a column for each option in the scipy.signal.find_peaks() function. If you do not wish to define certain parameters, simply enter None for the parameter value (or 1 for the distance parameter).') 
    
    #iterate over all the columns in the csv (ie parameters)
    for column in df_strNone:
        
        #check that each column name is valid (there are no names that don't appear in the list of expected columns)
        if column not in list_expected_cols:
            raise Exception(str(column) + ' is not a valid column name in the peak_detection_parameter_csv. Please only use column names corresponding to the options for scipy.signal.find_peaks() function. Note that LibreOffice codes underscores as +AF8-, so column names containing underscores may need to be fixed.')
        
        #whenever the parameter is a string, evaluate it literally to obtain Nonetype or tuple or integer
        if type(df_strNone[column][0]) == str:
            dict_trueNone[column] = ast.literal_eval(df_strNone[column][0])
        else:
            dict_trueNone[column] = df_strNone[column][0]
                
    return dict_trueNone

def extract_all_pulseox_metrics(analysis_type, input_trace, tot_length_seconds, output_name, image_output_type,
                                peak_detection_parameter_csv, invert_bool, window_length,
                                fMRI_censoring_mask_csv, fMRI_TR, large_window_width, large_window_overlap):
    '''This function combines all the other functions in order.'''
    raw_trace_arr = pd.read_csv(input_trace, header = None)[0]
    tot_num_samples = len(raw_trace_arr)
    sampling_rate = int(tot_num_samples/tot_length_seconds)
    time_array = np.arange(start=0, stop=tot_length_seconds , step=1/sampling_rate)
    basename = os.path.splitext(os.path.basename(input_trace))[0]
    output_name = output_name + "/" + basename + "_"
    
    ####################################### Check user inputs ######################################
    if peak_detection_parameter_csv is not None:
        param_dict = process_user_csv(peak_detection_parameter_csv)
    else:
        param_dict = {'height': None, 'threshold':None, 'distance': 45, 'prominence': 10, 'width': 50, 'wlen': None, 
                      'rel_height': 1, 'plateau_size': (1,200)}
        
    if fMRI_censoring_mask_csv is not None:
        #the censoring df is generated from the EPI, so it has length tot_length_seconds 
        # multiply it by sampling rate to get array of length tot_num_samples - convert so True represents points that ARE censored
        fMRI_censoring_df = pd.read_csv(fMRI_censoring_mask_csv)
        censoring_arr_full = np.repeat(np.array(fMRI_censoring_df), int(sampling_rate*fMRI_TR)) == False
        indices_of_censored_samples = np.where(censoring_arr_full ==1)[0]
    else:
        censoring_arr_full = None
    ######################################### PREPROCESSING #####################################
    #denoise
    trace_smoothed = denoise(raw_trace_arr, sampling_rate, invert_bool)
    del raw_trace_arr
    gc.collect()
    
    ######################################### WAVELET #####################################
    if analysis_type == 'wavelet_only':
        wavelet, wavelet_norm = get_wavelet(trace_smoothed, sampling_rate, time_array,tot_num_samples, output_name, image_output_type)
        
    ######################################### PEAK DETECTION #####################################
    elif (analysis_type == 'peak_detection_only') | (analysis_type == 'compute_metrics'):
        #extract the heart beat indices
        beat_indices, beats_bool, beats_toplot, beat_prominence = find_heart_beats(trace_smoothed, param_dict)
        if fMRI_censoring_mask_csv is not None:
            #find the location of censored breaths within the breath_indices_window
            location_of_censored = np.where(np.isin(beat_indices, indices_of_censored_samples))
        #resp rate in rolling window - per sample
        HR_inst, HR_inst_censored = get_HR_inst(beats_bool, censoring_arr_full, window_length, sampling_rate)
        
        ###################### PEAK DETECTION ONLY -PLOT PEAKS FOR QC ###########################
        if analysis_type == 'peak_detection_only':
            samples_per_iteration = int(sampling_rate*20)
            start = 0
            end = samples_per_iteration
            while end < tot_num_samples: 
                fig, ax = plt.subplots(figsize = (12,4))
                #plot the respiration trace and the detected breaths to make sure that they were properly detected
                #offset the trace by 60 so that there's less white space
                ax.plot(time_array[start:end], trace_smoothed[start:end]+60, label = 'Smoothed Plethysmography Trace')
                ax.plot(time_array[start:end], beats_toplot[start:end]+ 60, '.', label = 'Detected Heart Beats')
                ax.plot(time_array[start:end], HR_inst[start:end], label = 'Heart Rate (bpm)')
                if fMRI_censoring_mask_csv is not None:
                    ax.fill_between(time_array[start:end], 0, 1, where=censoring_arr_full[start:end], facecolor='red', alpha=0.2,
                                    transform=ax.get_xaxis_transform())
                ax.set_xlabel('Time (s)')
                ax.set_title('Quality Control Heart Beat Detection')
                ax.legend(ncol=3)
                if image_output_type == 'svg':
                    fig.savefig(output_name + 'QC_pleth-start_' + str(int(time_array[start])) + 's.svg')
                else:
                    fig.savefig(output_name + 'QC_pleth-start_' + str(int(time_array[start])) + 's.png')
                plt.close()
                start = start + samples_per_iteration
                end = end + samples_per_iteration
            #save outputs
            if fMRI_censoring_mask_csv is not None:
                df_basic = pd.DataFrame({'HR_censored': HR_inst_censored})     
                df_basic.to_csv(output_name + "HR_censored.csv")
            else:
                df_basic = pd.DataFrame({'HR': HR_inst})     
                df_basic.to_csv(output_name + "HR.csv")       
        ######################################### COMPUTE METRICS ######################
        if analysis_type == 'compute_metrics':
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
            
            #extract entropy in a 5s window 
            with warnings.catch_warnings():
                warnings.simplefilter(action = "ignore", category = RuntimeWarning)
                entropy_inst, entropy_inst_full_length = get_entropy_in_inst_window(trace_smoothed, censoring_arr_full, 5,
                                                                                    sampling_rate, tot_num_samples)

            ################## SAVE OUTPUTS ################
                        #for the metrics where there is only one value per breath, repeat same value until next breath
            period_btw_beats_persample = repeat_values_for_plotting(period_btw_beats, beats_bool, beat_indices)
            hrv_inst_std_period_persample = repeat_values_for_plotting(hrv_inst_std_period, beats_bool, beat_indices)
            hrv_inst_rmssd_period_persample = repeat_values_for_plotting(hrv_inst_rmssd_period, beats_bool, beat_indices)
            pvi_inst_persample = repeat_values_for_plotting(pvi_inst, beats_bool, beat_indices)

            df_persample = pd.DataFrame({'Pulseox_trace_smoothed': trace_smoothed, 'Beats': beats_toplot, 'HR': HR_inst, 'Period': period_btw_beats_persample, 
                                        'HRV_std': hrv_inst_std_period_persample, 'HRV_rmssd': hrv_inst_rmssd_period_persample, 'PVI': pvi_inst_persample, 'Entropy': entropy_inst_full_length})     
            df_persample.to_csv(output_name + "pleth_metrics_per_sample.csv")

            ######################################### EXTRACT AVERAGE METRICS IN WINDOW ##################
        if (analysis_type == 'compute_metrics') & (large_window_width is not None):
            #if no overlap is provided, set it to 0
            if large_window_overlap is None:
                large_window_overlap = 0
                #create arrays to store the values for for all windows
                num_windows = int(tot_length_seconds/large_window_width) 
            else:
                num_windows = 1+int((tot_length_seconds - large_window_width)/large_window_overlap) #numerator gives last start, frac gives num starts
            metrics_in_window = np.zeros((num_windows,15))

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
                beat_indices_window = np.copy(beat_indices_window_nocensor)
                if fMRI_censoring_mask_csv is not None:
                    beat_indices_window[location_of_censored] = False

                #extract mean/std of instantaneous HR in that window
                if fMRI_censoring_mask_csv is not None:
                    metrics_in_window[count,2] = np.nanmean(HR_inst_censored[large_window_start_samplenum:large_window_end_samplenum])
                    metrics_in_window[count,3] = np.nanstd(HR_inst_censored[large_window_start_samplenum:large_window_end_samplenum])
                else:
                    metrics_in_window[count,2] = np.nanmean(HR_inst[large_window_start_samplenum:large_window_end_samplenum])
                    metrics_in_window[count,3] = np.nanstd(HR_inst[large_window_start_samplenum:large_window_end_samplenum])

                #extract mean of instantaneous HRV in that window
                metrics_in_window[count,4] = np.mean(hrv_inst_std_period[beat_indices_window])
                metrics_in_window[count,5] = np.mean(hrv_inst_rmssd_period[beat_indices_window])

                #extract mean of instantaneous pvi in that window
                metrics_in_window[count,6] = np.mean(pvi_inst[beat_indices_window])

                #extract mean of instantaneous entropy in that window
                metrics_in_window[count,7] = np.mean(entropy_inst_full_length[large_window_start_samplenum:large_window_end_samplenum])

                #extract overall resp rate across whole window (can't censor breaths or will give inaccurate rate
                metrics_in_window[count,8] = (60/large_window_width)*(beat_indices[beat_indices_window_nocensor].size)

                #extract mean period and variability in period (RRV) across whole window
                metrics_in_window[count,9] = np.mean(period_btw_beats[beat_indices_window])
                metrics_in_window[count,10] = np.std(period_btw_beats[beat_indices_window])
                metrics_in_window[count,11] = np.mean(period_ssd[beat_indices_window])**(1/2)

                #extract mean widths in that window
                metrics_in_window[count,12] = np.mean(width_quart[beat_indices_window])
                metrics_in_window[count,13] = np.mean(width_half[beat_indices_window])
                metrics_in_window[count,14] = np.mean(width_base[beat_indices_window])

                #NOTE: I am NOT extracting the entropy across the whole window due to prohibitive memory/time constraints - also preliminary         
                #analysis showed that entropy across the whole window is very highly correlated with mean of instantaneous entropy.

                #set the start time of the next window in realtime
                large_window_start_realtime = large_window_end_realtime - large_window_overlap
                count = count+1

            ######################################## SAVE OUTPUTS ######################################
            df_onesample_per_window = pd.DataFrame(metrics_in_window, columns = ['Window start time','Window end time','Instantaneous HR-window mean', 'Instantaneous HR - window std', 'Instantaneous HRV period std-window mean','Instantanous HRV period rmssd-window mean', 'Instantaneous PVI-window mean','Instantaneous entropy-window mean','HR-overall window','Period-overall window mean', 'HRV-overall period window std', 'HRV-overall period window rmssd', 'Width at quarter height-overall window mean', 'width at half height-overall window mean','Width at base-overall window mean'])
            df_onesample_per_window.to_csv(output_name + "pleth_metrics_per_window.csv")
    else:
        raise Exception('analysis_type is not valid. Please enter one of the three options: wavelet_only, peak_detection_only or compute_metrics.')

###################################################### CALL FUNCTION ##################################
extract_all_pulseox_metrics(analysis_type, input_trace, tot_length_seconds, output_name, image_output_type,
                                peak_detection_parameter_csv, invert_bool, window_length,
                                fMRI_censoring_mask_csv, fMRI_TR, large_window_width, large_window_overlap)
gc.collect()
    

