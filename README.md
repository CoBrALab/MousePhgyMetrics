# MousePhgyMetrics

# **Purpose**
This tool was designed for processing raw respiration and plethysmography traces obtained from a mouse. It performs the following steps:
1) Smoothes the raw trace to reduce noise
2) Extracts peaks corresponding to breaths or heart beats
3) Calculates various instantaneous metrics based on the peaks, such as respiration rate, heart rate, heart rate variability etc. 
4) Outputs the wavelet transform of the trace for quality control purposes.

# **Installation** 
Download from github into the desired folder:
`git clone https://github.com/CoBrALab/MousePhgyMetrics`

Create the environment from the provided environment file:`conda env create --file environment.yml`
# **Use**
1. `module load anaconda`
2. `conda activate phgy_analysis`
3. `python`
3. ``` 
    import respiration_functions
    import cardiac_functions
    ```
    (run from the folder containing respiration_functions.py and cardiac_functions.py)
    
# **Functions** 
Respiration:
```
respiration_functions.extract_all_resp_metrics(raw_resp_trace_arr, large_window_width, large_window_overlap, window_length, tot_num_samples,
                             tot_length_seconds, output_name, invert_bool, h, d, pr, pl_min, pl_max, wl, 
                             CENSOR_bool, df_censoring, QC_WAVELET_PEAK_bool, QC_PEAK_ONLY_bool, ALL_MEASURES_bool, 
                             ALL_MEASURES_WINDOW_bool)
 <raw_resp_trace_arr> : csv containing a single column of the raw respiration trace samples
 --large_window_width: width of window (in seconds) if you wish to compute the average respiration metrics within a window. Otherwise, put None.
 --large_window_overlap: overlap between windows (in seconds). Otherwise, put None.
 --window_length: rolling window length (in seconds) over which to smooth respiration trace. Default is 2.
 --tot_num_samples: the total number of samples of the raw respiration trace.
 --tot_length_seconds: the total duration in seconds of the raw respiration trace.
 --output_name: path and name of the output files
 --invert_bool: whether or not the respiration trace should be inverted (e.g. if the respiration pillow was placed upside down). Either True or False.
 --h: height parameter controls peaks that are detected. Sets minimum peak height. Default 0.25.
 --d: distance parameter controls peaks that are detected. Sets minimmum distance in number of samples between peaks. Default 55.
 --pr: prominence parameter controls peaks that are detected. Sets vertical distance between peak and trough. Default 3.
 --pl_min, pl_max: plateau parameters control peaks that are detected. Set min and max plateau width in number of samples. Defaults are 1, 30.
 --wl: width parameter controls peaks that are detected. Default None.
 --CENSOR_bool: boolean determining whether or not respiration trace should be censored according to the fMRI censoring array.
 --df_censoring: dataframe of the fMRI censoring array produced by RABIES.
 --QC_WAVELET_PEAK_bool: if True, outputs both the wavelet and preliminary peaks for quality control. For use the first time the function is applied.
 --QC_PEAK_ONLY_bool: if True, outputs only the peaks but doesn't recompute wavelet. For efficient tuning of peak detection parameters.
 --ALL_MEASURES_bool: if True, outputs all measures (entropy, RRV, period, periodicity). For use after optimal peak detection parameters have been settled.
 --ALL_MEASURES_WINDOW_bool: if True, computes the average of all measures within the specified time window.
 ```
 Pulse oximetry:
 ```
cardiac_functions.extract_all_pulseox_metrics(raw_trace_arr, large_window_width, large_window_overlap, window_length, tot_num_samples,
                             tot_length_seconds, output_name, h, d, pr, pl_max, width_val, wavelet_band_width,
                             wavelet_peak_dist, num_bands_to_detect, invert_bool, num_to_avg, CENSOR_bool,
                             df_censoring, QC_WAVELET_PEAK_bool, QC_PEAK_ONLY_bool, ALL_MEASURES_bool, 
                             ALL_MEASURES_WINDOW_bool)
 <raw_trace_arr> : csv containing a single column of the raw pulse oximetry trace samples
 --large_window_width: width of window (in seconds) if you wish to compute the average metrics within a window. Otherwise, put None.
 --large_window_overlap: overlap between windows (in seconds). Otherwise, put None.
 --window_length: rolling window length (in seconds) over which to smooth trace. Default is 1.
 --tot_num_samples: the total number of samples of the raw trace.
 --tot_length_seconds: the total duration in seconds of the raw trace.
 --output_name: path and name of the output files
 --h: height parameter controls peaks that are detected. Sets minimum peak height. Default None.
 --d: distance parameter controls peaks that are detected. Sets minimmum distance in number of samples between peaks. Default 45.
 --pr: prominence parameter controls peaks that are detected. Sets vertical distance between peak and trough. Default 10.
 --pl_max: plateau parameters control peaks that are detected. Set max plateau width in number of samples. Defaults is 200.
 --width_val: width parameter controls peaks that are detected. Default 50.
 --wavelet_band_width: parameter to control identification of unique frequency bands in the wavelet. Default 0.5 seconds.
 --wavelet_peak_dist: parameter to control identification of unique frequency bands in the wavelet. Default 8. 
 --num_bands_to_detect:parameter to control identification of unique frequency bands in the wavelet. Default 2.  
 --invert_bool: whether or not the trace should be inverted. Default False.
 --num_to_avg: parameter to control identification of unique frequency bands in the wavelet. Default 10 000. 
 --CENSOR_bool: boolean determining whether or not respiration trace should be censored according to the fMRI censoring array.
 --df_censoring: dataframe of the fMRI censoring array produced by RABIES.
 --QC_WAVELET_PEAK_bool: if True, outputs both the wavelet and preliminary peaks for quality control. For use the first time the function is applied.
 --QC_PEAK_ONLY_bool: if True, outputs only the peaks but doesn't recompute wavelet. For efficient tuning of peak detection parameters.
 --ALL_MEASURES_bool: if True, outputs all measures (entropy, RRV, period, periodicity). For use after optimal peak detection parameters have been settled.
 --ALL_MEASURES_WINDOW_bool: if True, computes the average of all measures within the specified time window.
 ```
