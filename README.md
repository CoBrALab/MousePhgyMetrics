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
