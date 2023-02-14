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

# **Usage** 

## Complete Command line interface
```
Usage: execute_analysis.sh [--output_path <arg>] [--output_image_type <arg>] [--peak_detection_parameter_csv <arg>] [--window_length <arg>] [--fMRI_censoring_mask_csv <arg>] [--fMRI_TR <arg>] [--average_metrics_window_length <arg>] [--average_metrics_window_overlap <arg>] [-h|--help] <data_type> <analysis_type> <input_trace> <tot_length_seconds>
        <data_type>: Specify either respiration or plethysmography - will determine the processing workflow and metrics extracted
        <analysis_type>: Specify one of three options: wavelet_only, peak_detection_only, compute_metrics. See README for details.
        <input_trace>: The raw physiological recording, must be a 1D csv file.
        <tot_length_seconds>: The exact total duration of the provided physiological recording, in seconds.
        --output_path: Prefix to the output csvs and image files. Must not already exist. (default: './physiology_analysis_outputs')
        --output_image_type: Type of image to output, either svg (for publication-level quality) or png (default: 'png')
        --peak_detection_parameter_csv: Parameters that determine the which peaks in the trace are counted as breaths/heart beats. Refer to the scipy.signal.find_peaks() documentation for the full list of parameters. See README for the default values for each data type. (no default)
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
 
 #### Example command
`bash execute_analysis.sh respiration peak_detection_only /home/sub-001_ses-1_acq-respiration_run-1.txt 1440 `
or

`bash execute_analysis.sh plethysmography peak_detection_only /home/sub-001_ses-1_acq-pulseox_run-1.txt 1440 `

#### Example output
The function will output an image of the smoothed respiratory/plethymography trace, with the labelled peaks and resulting respiration/heart rate calculated from the peaks. There is a separate image for every 20-30 s of data. A csv with the respiration/heart rate is also produced.
![QC_pleth-start_1200s](https://user-images.githubusercontent.com/47565996/218807749-390b49d0-9a8e-4b38-b0f2-de865affc02f.png)

 ## Step 3: QC the breaths and/or heart beats
 
