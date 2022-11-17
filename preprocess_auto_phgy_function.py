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
from scipy import stats

def phgy_preprocess(phgy_df, num_columns, outlier_min, outlier_max):
    
    #replace -1 with Nan (to be able to interpolate values)
    phgy_df_nan = phgy_df.replace([-1,0], np.nan)
    
    #interpolate values, and count how many were missing
    phgy_df_interpolate = phgy_df_nan.interpolate()
    
    num_nan = phgy_df_nan.isna().sum()
    
    #remove outliers separately for each column
    outlier_count = np.zeros(num_columns)
    for col in range(0,num_columns):
        q_low = phgy_df_interpolate.iloc[:,col].quantile(outlier_min)
        q_hi  = phgy_df_interpolate.iloc[:,col].quantile(outlier_max)
        
        bool_arr = ((phgy_df_interpolate.iloc[:,col] < q_hi) & (phgy_df_interpolate.iloc[:,col] > q_low))
        outlier_indices = phgy_df_interpolate.index[bool_arr == False].tolist()
        
        phgy_df_interpolate.iloc[outlier_indices,col] = np.nan
        outlier_count[col] = len(phgy_df_interpolate.iloc[outlier_indices,col])
        
    #interpolate values again to replace the outliers
    phgy_df_interpolate2 = phgy_df_interpolate.interpolate()
    
    #crop the first few seconds, and the remaining time after 1440 (means acq was stopped late)
    phgy_df_finaltime = phgy_df_interpolate2[2:1442]
    
    return phgy_df_finaltime