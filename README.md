# MousePhgyMetrics

# **Purpose**
This tool was designed for processing raw respiration and plethysmography traces obtained from a mouse. It performs the following steps:
a) Smoothes the raw trace to reduce noise
b) Outputs the wavelet transform of the trace for quality control purposes.
c) Extracts peaks corresponding to breaths or heart beats
d) Calculates various instantaneous metrics based on the peaks, such as respiration rate, heart rate, heart rate variability etc. 

# **Installation** 
Download from github into the desired folder:
`git clone https://github.com/CoBrALab/MousePhgyMetrics`

Create the environment from the provided environment file:`conda env create --file environment.yml`

# **Usage** 

## Complete Command line interface
```
Usage: execute_analysis.sh [--output_path <arg>] [--output_image_type <arg>] [--peak_detection_parameter_csv <arg>] [--invert_trace_boolean <arg>] [--window_length <arg>] [--fMRI_censoring_mask_csv <arg>] [--fMRI_TR <arg>] [--average_metrics_window_length <arg>] [--average_metrics_window_overlap <arg>] [-h|--help] <data_type> <analysis_type> <input_trace> <tot_length_seconds>
        <data_type>: Specify either respiration or plethysmography - will determine the processing workflow and metrics extracted
        <analysis_type>: Specify one of three options: wavelet_only, peak_detection_only, compute_metrics. See README for details.
        <input_trace>: The raw physiological recording, must be a 1D .csv or .txt file.
        <tot_length_seconds>: The exact total duration of the provided physiological recording, in seconds.
        --output_path: Prefix to the output csvs and image files. Must not already exist. (default: './physiology_analysis_outputs')
        --output_image_type: Type of image to output, either svg (for publication-level quality) or png (default: 'png')
        --peak_detection_parameter_csv: Parameters that determine the which peaks in the trace are counted as breaths/heart beats. Refer to the scipy.signal.find_peaks() documentation for the full list of parameters. See README for the default values for each data type. (default: 'None')
        --invert_trace_boolean: Whether the raw trace should be inverted (switch peaks and throughs). (default: 'False')
        --window_length: rolling window length (in seconds) over which to smooth respiration trace. (default: '2')
        --fMRI_censoring_mask_csv: If comparing to censored fMRI data, provide the 1D csv of boolean values for each fMRI timepoint, and the physiological outputs will be censored according to the same csv, so that high motion timepoints are excluded from the analysis. (default: 'None')
        --fMRI_TR: If providing the fMRI_censoring_mask_csv, specify also the TR with which the fMRI data was acquired, in seconds. (default: '1.0')
        --average_metrics_window_length: If wish to compute the average of each metric in a time window, choose the window length (e.g 120s). Only executes when analysis_type=compute_metrics (default: 'None')
        --average_metrics_window_overlap: If wish to compute the average of each metric in a rolling time window, choose the overlap between windows (e.g 60s). Only executes when analysis_type=compute_metrics (default: 'None')
        -h, --help: Prints help 
```
 
 ## Step 0: Activate the environment
 If running from the CIC, this will look like:
 ```
module load anaconda
source activate phgy_analysis
```
 
 ## Step 1: Obtain wavelet transform
 
 The wavelet transform is a data-driven approach to obtain the spectral (frequency) properties of the input trace. It provides an estimate of the power across frequencies at each timepoint and is robust to noise. Thus, by examining at the wavelet transform of the respiration trace, you will be able to see the instantaneous respiration rate for each sample. By examining the wavelet transform of the plethysmography trace, you will see frequency bands for both the respiration rate and heart rate. This is convenient for getting a quick overview of your data and for performing quality control of later calculations that rely on detecting peaks in the data. NOTE: the wavelet transform is not typically used to extract values of respiration/heart rate and cannot provide other measures such as PVI.

 #### Example command
`bash execute_analysis.sh respiration wavelet_only /home/sub-001_ses-1_acq-respiration_run-1.txt 1440 `
or

`bash execute_analysis.sh plethysmography wavelet_only /home/sub-001_ses-1_acq-pulseox_run-1.txt 1440 `

#### Example output
Below is the wavelet transform of a plethysmography trace. The strong red/yellow frequency band at 200-300 bpm corresponds to the heart rate whereas the secondary cyan frequency band at 100 bpm corresponds to the respiration rate (which is also captured in the plethysmography recording albeit less clearly than in a respiratory recording). You can easily see that both heart rate and respiration rate increase slightly across time (because we changed the isoflurane dose across time in this experiment) and visually observe the rate at any given time.
![respiration_wavelet_transform](https://user-images.githubusercontent.com/47565996/218801029-4bbb29ba-0e66-4e8b-9c8f-c1db9bdc10bb.png)

 ## Step 2: Detect the breaths and/or heart beats using the default parameters
 
 The breaths and/or heart beats are identified using the scipy.signal.find_peaks() function applied on the respiration and plethysmography traces respectively. This function will detect all peaks that have a certain absolute height, relative hieght, width, vertical and horizontal distance from neighboring peaks. We have set default values for the aforementioned parameters that should result in accurate peak detection for most mouse data. However, under abnormal conditions or in certain mice, the default values won't be appropriate and will result in either under-detection (peaks are skipped and not labelled as breaths when they should be) or over-detection (noise-related fluctuations are incorrectly labelled as breaths). Thus, we recommend first running the peak_detection step with the default values, quality-controlling the outputs, and changing the default parameters to improve peak detection in certain subjects if necessary. 

 The default detection values for breaths in the respiration trace are:
 'height': 0.25, 'threshold':None, 'distance': 55, 'prominence': 3, 'width': None, 'wlen': None, 'rel_height': None, 'plateau_size': (1,30).

 The default detection values for heart beats in the plethysmography trace are:
 'height': None, 'threshold':None, 'distance': 45, 'prominence': 10, 'width': 50, 'wlen': None,'rel_height': 1, 'plateau_size': (1,200).
 
 #### Example command
 In the following example, we also provide the temporal censoring csv for simultaneously acquired fMRI data. See the file sample_fMRI_censoring_mask.csv for how the csv should be formatted - we used the csvs produced by the RABIES software.
 
`bash execute_analysis.sh respiration peak_detection_only /home/sub-001_ses-1_acq-respiration_run-1.txt 1440 --fMRI_censoring_mask_csv /rabies_outputs/sub-001_ses-1_task-rest_acq-EPI_run-1_bold_RAS_combined_frame_censoring_mask.csv --fMRI_TR 1.0`
or

`bash execute_analysis.sh plethysmography peak_detection_only /home/sub-001_ses-1_acq-pulseox_run-1.txt 1440 --fMRI_censoring_mask_csv /rabies_outputs/sub-001_ses-1_task-rest_acq-EPI_run-1_bold_RAS_combined_frame_censoring_mask.csv --fMRI_TR 1.0`

#### Example output
The function will output an image of the smoothed respiratory/plethymography trace, with the labelled peaks and resulting respiration/heart rate calculated from the peaks. There is a separate image for every 20-30 s of data. A csv with the respiration/heart rate is also produced. Any timepoints that were censored from the fMRI data due to high motion are indicated in red.
![QC_pleth-start_0s](https://user-images.githubusercontent.com/47565996/218831448-37d8ecfc-d367-49fd-b894-e6799e5f6f48.png)

 ## Step 3: QC the breaths and/or heart beats
 
 Examine all the png files generated. To identify incorrectly labelled or skipped peaks, check that the estimated rate (green) matches the frequency from the wavelet transform. In general, the rate should stay relatively consistent across the 20-30s span, so any sudden changes in the rate may indicate that peaks were incorrectly labelled. 
 
 #### 'Before' example
 With the default parameters, certain peaks that are smaller than usual were not labelled (red arrows). 
 
 ![image](https://user-images.githubusercontent.com/47565996/218837174-d33ac850-720a-4b79-9524-4c1a8eba6daf.png)

To fix this, re-run the peak_detection_only step but specify a custom peak_detection_parameter_csv (see the sample provided). In this case, we decreased the prominence of the peak detection (see the scipy.signal.find_peaks() documentation to understand the purpose of each parameter https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

`bash execute_analysis.sh plethysmography peak_detection_only /home/sub-001_ses-1_acq-pulseox_run-1.txt 1440 --peak_detection_parameter_csv /home/sub-001_ses-1_pulseox_peak_detection_parameters.csv`

 #### 'After' example
 With the adjusted parameters, most peaks are now correctly labelled - notice how the heart rate at 268 s is now consistent across time, and it also matches the frequency from the wavelet transform. Nevertheless, one peak is still being skipped; due to its very small size, it will likely be difficult to detect (if the prominence is lowered further, the algorithm may pick up on other peaks that are not heart beats). Thus, if there is occasionally one beat missed or over-detected, it may not be worth tuning the parameters, particularly if you want the average HR across 1 whole minute. This decision depends on the goal of the analysis and the nature of the data (e.g. how noisy it is). 
 
 ![QC_pleth-start_260s](https://user-images.githubusercontent.com/47565996/218838752-2ab99173-e914-4e02-9693-321ae8cbaf2d.png)
 
  ## Step 4: Compute additional metrics (if desired)
  Once you are confident in the peak detection parameters, you can use the HR.csv or RR.csv generated by the peak_detection_only step for your analysis, or you can also compute additional metrics, such as heart rate variability. 
  
  #### Example command
  In this example, we also specify that we want to compute the average metrics over a rolling window of 120s where consecutive windows overlap by 60s.
 
 `bash execute_analysis.sh respiration compute_metrics /home/sub-001_ses-1_acq-respiration_run-1.txt 1440 --average_metrics_window_length 120 --average_metrics_window_overlap 60`
or

`bash execute_analysis.sh plethysmography compute_metrics /home/sub-001_ses-1_acq-pulseox_run-1.txt 1440 --average_metrics_window_length 120 --average_metrics_window_overlap 60` 

  #### Example outputs
  The full list of metrics for each modality is specified below.
  
  Plethysmography: heart rate (HR), period, heart rate variability computed from the standard deviation of the period (HRV_std), heart rate variability computed from the root mean square of successive differences between normal heartbeats (HRV_rmssd), plethysmography variability index (PVI), entropy and metrics that characterize the heart beat shape (width at base, width at half max and width at quarter max).
  
  Respiration: respiration rate (RR), period, respiratory variability (RV) computed from the standard deviation of the detrended-smoothed-respiration trace, resp rate variability (RRV) computed from the standard deviation of the period (RRV_std), resp rate variability computed from the root mean square of successive differences between normal breaths (RRV_rmssd) and entropy.
 
 The output file pleth_metrics_per_sample.csv will contain an instantaneous value of the aforementioned metrics for each sample of the original trace. As the period, HRV and PVI are calculated from beat to beat, the value for these metrics will only change with each new breath. There will be a gap of a few seconds at the start of each csv.
 
  The output file pleth_metrics_per_window.csv is produced when the user specifies a window length to average over. The file contains the average of each instantaneous metric over the window (e.g. "Instantaneous HRV period std-window mean") as well as the result of computing that metric directly over the whole window (e.g. "HRV-overall period window std" is the standard deviation of all periods across the whole window). The two results are often very highly correlated, but both are provided anyways.
