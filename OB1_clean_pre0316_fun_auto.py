'''
---
# **CHAPTER 3: OB1 Clean Pre0316 Auto**
---

**Pre 2023-03-16-P05 (_0221-P01, 0314-P02, 0314-P03, 0315-P04_)**  
-  (Y ← OP_X | Z ← OP_Y | X ← OP_Z )

**Post 2023-04-02**
- Check Normality
  - Return **op_highest_peak, ma_highest_peak** from functions **5) Analyze Data**...
  - 1-3) Angle --> min_angle() + max_angle()
  - 1-4) Reach --> peaks_method2() --> reach_lists()
  - 1-5) Speed --> speed_max()

- Store OP Data Tracking Accuracy
  - track_op() --> tracking csv file

**Fixed on 2023-04-11**
- Problem 1: Losing Complete Sight of the User

**Tweaked on 2023-05-22**
- "_**Load automatic peak values to clean**_"
- def synch_op(op_synch, op_thresh, op_dist, op_peak, op_end)
- def synch_ma(ma_synch, op_synch, ma_thresh, ma_dist, ma_peak)

**5 Bootle Blast + 18 Boot Camp**
- (1-1) Joint Coordinate Position 
- (1-2) Body Segment Length
- (1-3) Joint Angle ROM

**5 Bootle Blast**
- (1-4) Extent of Hand Reach
- (1-5) Max/Mean Hand Speed

Soowan Choi
'''

# 1 - Import Libraries
# to clean and analyze data
import pandas as pd
import numpy as np
import math
import re
import pylab
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import norm
from scipy.stats import kstest
from scipy.stats import probplot
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

# to resampled data
from scipy import signal
from sklearn.utils import resample

# to visualize data
import matplotlib.pyplot as plt

# to standardize data
from sklearn.preprocessing import StandardScaler

# to find peaks in data for synchronization
from scipy.signal import find_peaks


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def remove_outliers(array):
  #calculate interquartile range 
  q1, q3 = np.percentile(array, [25 ,75])
  iqr = q3 - q1

  lower = q1 - 1.5*iqr
  upper = q3 + 1.5*iqr

  lower_outliers = len(array[array < lower])
  upper_outliers = len(array[array > upper])

  outliers_index = list(array.index[array < lower].values) +  list(array.index[array > upper].values)

  # drop upper and lower outliers
  print(f'Lower Outliers: {lower_outliers} Upper Outliers: {upper_outliers}')
  array = array.drop(outliers_index)

  return array


def clean_op(op_clean):
  '2.2.1) Clean Dataframe'

  print('\n2.2.1) CLEAN DATAFRAME\n----------------------------\n')

  # skip the first empty row
  op_clean = op_clean.drop(0,axis=0) 

  # reset index number
  op_clean = op_clean.reset_index().drop('index',axis=1)  

  # convert entire DataFrame string to float
  print(f'Current Data Types: {op_clean.dtypes.unique()}\n')
  op_clean = op_clean.astype(float)
  print(f'Fixed Data Types: {op_clean.dtypes.unique()}')

  # convert OP from m -> cm
  op_clean[op_clean.select_dtypes(include=['number']).columns] *= 100

  # change time from milliseconds to seconds
  op_clean.Time /= 100000
  op_clean.Time = op_clean.Time - op_clean.Time.iloc[0]

  # what is the orbbec frequency?
  frames = len(op_clean.Time)
  seconds = (op_clean.Time.iloc[-1] - op_clean.Time.iloc[0]) 
  op_hz = frames /seconds
  print(f'\nNotice {frames} frames in {seconds} seconds = {op_hz} Hz for Orbbec! \n')

  return op_clean, op_hz


def coord_op(op_coord):
  '2.2.2) Coordinate Transformation (Y <-- OP_X | Z <-- OP_Y | X <-- OP_Z)'

  print('\n2.2.2) COORDINATE TRANSFORMATION (Y <-- OP_X | Z <-- OP_Y | X <-- OP_Z)\n----------------------------\n')

  # remove the X Y Z indicators from column names
  op_coord.columns = op_coord.columns.str.replace('X$','',regex =True)
  op_coord.columns = op_coord.columns.str.replace('Y$','',regex =True)
  op_coord.columns = op_coord.columns.str.replace('Z$','',regex =True)

  # create list of new column names
  naming_system = ['Y','Z','X']  # list the specific column measurement names to add to new column
  op_tmp = op_coord.columns.copy()     # create a copy of the column name order
  new_col = []                   # empty list to store new column names
  for i in range(1, 57, 3):      # loop from 'Head' index to index of last relevant column, increment every 3 steps identical to csv file
    for j in range(0,3):
      op_coord.rename(columns = {op_coord.columns[i+j]:op_tmp[i+j] + naming_system[j]}, inplace= True)    # change the specific name of the six succeeding columns in place
      new_col.append(op_coord.columns[i])

  # list of new column names  
  op_col = list(op_coord.columns)   # convert column names to a list (lists are mutable)
  for i in range(1,57):       # change relevant column names
    op_col[i] = new_col[i-1]  # change the column name using index

  # rename columns in data using list of new column names
  op_coord.columns = op_col

  return op_coord


def synch_op(op_synch, op_thresh, op_dist, op_peak, op_end):
  '2.2.4) Synchronization (Clapping Peak:Orbbec End Time)'

  print('\n2.2.4) SYNCHRONIZATION (CLAPPING PEAK:ORBBEC END TIME)\n----------------------------\n')

  # visualize OP data before synchronization point
  x = np.linspace(0,int(len(op_synch)),int(len(op_synch)))     # create timeframe using orbbec data length
  plt.plot(x,op_synch['WristLeftY'])        # plot the orbbec joint of interest
  plt.title(f'OP Clapping Motion WristLeftY - BEFORE SYNCHRONIZATION')
  plt.xlabel('Frames')
  plt.ylabel('Distance [cm]')
  plt.show()

  # zoom in on the peak movements
  # height = int(input("Minimal Peak Threshold: "))
  # distance = int(input("Minimal Distance Between Peaks (>1): "))
  height = op_thresh
  distance = op_dist
  x = op_synch['WristLeftY']
  peaks, _ = find_peaks(x, height= height, distance = distance)
  plt.plot(x)
  plt.plot(peaks, x[peaks], "x")
  plt.plot(np.zeros_like(x), "--", color="gray")
  plt.show()

  # show the peaks and corresponding index
  print('\nPeaks and Corresponding Index:')
  print(x[peaks])

  # locate the peak of the third clap
  # op_peak = int(input('\nIndex for peak of 3rd clap: '))
  op_peak = op_peak

  # locate the end frame of the game
  # end_frame_op = int(input('Orbbec Ending Frame (FROM GRAPH): '))
  end_frame_op = op_end

  print(f'\nShape of Orbbec BEFORE synchronization: {op_synch.shape}')
  op_synch = op_synch[op_peak:end_frame_op]       # cut orbbec data from starting position (horizontal peak) to ending position (frame at end of game log)
  print(f'Shape of Orbbec AFTER synchronization: {op_synch.shape} \n')

  # visualize OP data after synchronization point
  x = np.linspace(0,int(len(op_synch)),int(len(op_synch)))     # create timeframe using orbbec data length
  plt.plot(x,op_synch['WristLeftY'])        # plot the orbbec joint of interest
  plt.title(f'OP Clapping Motion WristLeftY - AFTER Synchronization')
  plt.xlabel('Frames')
  plt.ylabel('Distance [cm]')
  plt.show()

  return op_synch


def track_op(op_track):
  'REMOVE UNTRACKED OP'

  print('\nREMOVE UNTRACKED OP\n----------------------------\n')

  # reset index number
  op_track = op_track.reset_index().drop('index',axis=1)  

  # columns with UNTRACKED values BEFORE REMOVING UNTRACKED 
  nulls = []
  untracked = []
  print('Joints with UNTRACKED values BEFORE REMOVING UNTRACKED:')
  for tracked in op_track.columns[65:]:
    null = op_track[tracked].unique()
    if len(null) > 1:
      untracked.append(tracked)
      null_col = tracked.split('T')[0]
      for col in op_track.columns[:65]:
        if null_col in col:
          nulls.append(col)
  print(nulls)

  # Important columns with UNTRACKED values BEFORE REMOVING UNTRACKED
  isna = []
  important = ['Head','Shoulder','Elbow','Wrist','ASIS','Knee','Ankle']
  print('\nIMPORTANT Joints with UNTRACKED values BEFORE REMOVING UNTRACKED:')
  for col in nulls:
    for imp in important:
      if imp in col:
        isna.append(col)
  print(isna)
  print('\n')

  # count how many UNTRACKED values to clean in each important column
  track_col = [] # joints untracked
  track_ind = ['Accuracy [%]', ' Untracked [#]', 'Total [#]', 'Start Index', 'End Index']
  acc = []       # tracking accuracy
  unt = []       # total untracked
  tot = []       # total values
  s_ind = []     # starting index of untracked
  e_ind = []     # ending index of untracked

  untracked_imp = []
  for col in untracked:
    for imp in important:
      if imp in col:
        untracked_imp.append(col)
        values = op_track[col][op_track[col] == 0.0]
        count = len(values)
        percentage = round((len(op_track[col]) - count) / len(op_track[col]) * 100, 3)
        track_col.append(col)
        acc.append(percentage)
        unt.append(count)
        tot.append(len(op_track[col]))
        s_ind.append(values.index[0])
        e_ind.append(values.index[-1])
        print(f"{col} Column: {percentage}% TRACKING ACCURACY | {count} UNTRACKED Values | BETWEEN Index {values.index[0]} to {values.index[-1]}")

  track_data = [acc, unt, tot, s_ind, e_ind]
  tracking = pd.DataFrame(track_data, index = track_ind, columns = track_col)

  # for each joint, find index with null values and store in dictionary 
  untracked_dic = {}
  for col in untracked_imp:
      index_values = op_track[col][op_track[col] == 0.0]
      null_index = list(index_values.index.values)
      untracked_dic[col] = null_index

      # check if null appears at middle of data (instead of beginning or end)
      for i in range(len(null_index) - 1):
        if (null_index[i+1] - null_index[i]) != 1:
          print(f"{col}: Null Appears Middle of Data Around Index {null_index[i]}: {null_index[i-3:i+3]}")

  # create a list of overall untracked index to drop
  overall_list = []
  for col in untracked_dic:
    overall_list.append(untracked_dic[col])
  untracked_op_index = [unique_index for lists in overall_list for unique_index in lists]

  # only keep the rows with TRACKED values from important columns 
  print(f'\nshape of data BEFORE REMOVING UNTRACKED:{op_track.shape}')
  op_track = op_track.drop(op_track.index[untracked_op_index])
  print(f'shape of data AFTER REMOVING UNTRACKED:{op_track.shape}\n') 

  # columns with untracked values AFTER REMOVING UNTRACKED
  nulls = []
  untracked = []
  print('Joints with null values AFTER REMOVING UNTRACKED:')
  for tracked in op_track.columns[65:]:
    null = op_track[tracked].unique()
    if len(null) > 1:
      untracked.append(tracked)
      null_col = tracked.split('T')[0]
      for col in op_track.columns[:65]:
        if null_col in col:
          nulls.append(col)
  print(nulls)

  # Important columns with untracked values AFTER REMOVING UNTRACKED
  isna = []
  important = ['Head','Shoulder','Elbow','Wrist','ASIS','Knee','Ankle']
  print('\nIMPORTANT Joints with null values AFTER REMOVING UNTRACKED:')
  for col in nulls:
    for imp in important:
      if imp in col:
        isna.append(col)
  print(isna)
  print('\n')

  # count how many UNTRACKED values to clean in each important column
  untracked_imp = []
  for col in untracked:
    for imp in important:
      if imp in col:
        untracked_imp.append(col)
        values = op_track[col][op_track[col] == 0.0]
        count = len(values)
        print(f"{col} Column: {count} UNTRACKED Values | BETWEEN Index {values.index[0]} to {values.index[-1]} --> ({values.index[-1] - values.index[0]}) values?")

  return op_track, untracked_op_index, tracking 
  

def filte_op(op_filte, op_hz):
  '2.2.6) Filter Noise (Low-Pass Butterworth)'

  print('\n2.2.6) FILTER NOISE (LOW-PASS BUTTERWORTH)\n----------------------------\n')

  # save column names to rename new filtered data
  col_names = list(op_filte.columns)                
  time = op_filte['Time'].reset_index()
  del time['index']

  # filter all but the first time column and store filtered array data in a list
  empty = []                                   
  for col in op_filte.columns[1:]:
    y = op_filte[col]                           # for each column data
    y2 = butter_lowpass_filter(y,6,op_hz,4)     # filter the column data
    empty.append(y2)                            # store filtered data

  # create dataframe from stored filtered data & rename columns & reinsert time column
  op_filte = pd.DataFrame(empty).T                    
  op_filte.columns = col_names[1:]              
  op_filte.insert(0,'Time',time)                      
  
  return op_filte


def clean_ma(ma_clean):
  '3.2.1) Clean Dataframe'

  print('\n3.2.1) CLEAN DATAFRAME\n----------------------------\n')

  # drop first column (the frames) 
  ma_clean = ma_clean.drop("Frame#", axis = 1)

  # skip the first 2 rows
  ma_clean = ma_clean.iloc[2:,:]      

  # reset index number
  ma_clean = ma_clean.reset_index().drop('index',axis=1) 

  # convert entire DataFrame string to float
  print(f'Current Data Types: {ma_clean.dtypes.unique()}\n')
  ma_clean = ma_clean.astype(float)
  print(f'Fixed Data Types: {ma_clean.dtypes.unique()}')

  # convert MA from mm --> cm (except time)
  ma_clean.iloc[:,1:] = ma_clean.iloc[:,1:] / 10

  # what is the motion frequency?
  frames = len(ma_clean.Time)
  seconds = (ma_clean.Time.iloc[-1] - ma_clean.Time.iloc[0])  
  ma_hz = frames /seconds
  print(f'\nNotice {frames} frames in {seconds} seconds = {round(ma_hz,3)} Hz for Motion! \n')

  return ma_clean, ma_hz


def coord_ma(ma_coord):
  '3.2.2) Coordinate Transformation'

  print('\n3.2.2) COORDINATE TRANSFORMATION\n----------------------------\n')

  # isolate for relevant column names 
  new_names = []             
  for name in ma_coord.columns:
    if 'Unnamed' in name:     # irrelevant columns
      pass                 
    else:                     # relevant columns     
      new_names.append(name) 

  # search if relevant column exists in the dataframe and locate the index 
  dic = {}                                                     
  i = 0                                                             
  for name in new_names: 
    if name in ma_coord.columns:
      dic[i] = [name, 'index location:', ma_coord.columns.get_loc(name)]  # add index location of the column name to dictionary
      i += 1                                                        # increment dictionary index

  # rename columns of three succeeding index (to same name) from new_names list...
  naming_system = ['X','Y','Z']                     
  # get the index of the starting column name, which is 'Front.Head'
  start = ma_coord.columns.get_loc('Front.Head')            
  # create a copy of the column names
  ma_tmp = ma_coord.columns.copy()                    
  # loop from index of 'Front.Head' to index of last relevant column, increment every 3 steps (identical to csv file)    
  for i in range(start, dic[max(dic)][-1], 3):  
    # loop through the next three succeeding index
    for j in range(0,3):                            
      # change the specific name of the six succeeding columns in place
      ma_coord.rename(columns = {ma_coord.columns[i+j]:ma_tmp[i] + naming_system[j]}, inplace= True)

  return ma_coord       


def nullv_ma(ma_nullv):
  '3.2.3) INTERPOLATE NULL'

  print('\n3.2.3) INTERPOLATE NULL\n----------------------------\n')

  # columns with NULL values BEFORE INTERPOLATING 
  nulls = []
  print('Joints with NULL values BEFORE INTERPOLATING NULL:')
  for col in ma_nullv.columns:
    null = ma_nullv[col].isnull().unique()
    if len(null) > 1:
      nulls.append(col)
  print(nulls)

  # Important columns with NULL values BEFORE INTERPOLATING
  isna = []
  important = ['Front.Head','.ShoulderX','.ElbowX','.WristX','.ASISX','.KneeX','.AnkleX']
  print('\nIMPORTANT Joints with NULL values BEFORE INTERPOLATING NULL:')
  for col in nulls:
    for joint in important:
      if joint in col:
        isna.append(col)
  print(isna)
  print('\n')

  # count how many NULL values to clean in each important column
  for col in isna:
    #print(f"NULL Value Locations of the {col} Column:")
    count = 0
    for i,value in enumerate(list(ma_nullv[col])):
      if str(value) == 'nan':
        #print(i,value)
        count += 1
    print(f"{count} NULL Values in the {col} column")

  # find index with null values 
  null_index = []
  for col in isna:
    for index, val in enumerate(ma_nullv[col]):
      if str(val) == 'nan':
        null_index.append(index)

  # check if null appears at middle of data (instead of beginning or end)
  for i in range(len(null_index) - 1):
    if (null_index[i+1] - null_index[i]) != 1:
      print(f"Null Appears Middle of Data: {null_index[i-3:i+3]}")

  # interpolate and fill null values 
  ma_nullv = ma_nullv.interpolate(method = 'polynomial', order = 2, limit_direction = 'forward')

  # ***backwards fill AFTER initial interpolation, this data will be REMOVED later but required now as placeholders***
  ma_nullv = ma_nullv.interpolate(method = 'linear', limit_direction = 'backward')

  # columns with NULL values AFTER INTERPOLATING
  nulls = []
  print('Joints with NULL values AFTER INTERPOLATING NULL:')
  for col in ma_nullv.columns:
    null = ma_nullv[col].isnull().unique()
    if len(null) > 1:
      nulls.append(col)
  print(nulls)

  # IMPORTANT columns with NULL values AFTER INTERPOLATING 
  isna = []
  important = ['.ShoulderX','.ElbowX','.WristX','.ASISX','.KneeX','.AnkleX']
  print('\nIMPORTANT Joints with NULL values AFTER INTERPOLATING NULL:')
  for col in nulls:
    for joint in important:
      if joint in col:
        isna.append(col)
  print(isna)
  print('\n')

  # count how many NULL values to clean in each important column
  for col in isna:
    #print(f"NULL Value Locations of the {col} Column:")
    count = 0
    for i,value in enumerate(list(ma_nullv[col])):
      if str(value) == 'nan':
        #print(i,value)
        count += 1
    print(f"{count} NULL Values in the {col} column")

  return ma_nullv


def synch_ma(ma_synch, op_synch, ma_thresh, ma_dist, ma_peak):
  '3.2.4) Synchronization (Clapping Peak:Orbbec End Time)'

  print('\n3.2.4) SYNCHRONIZATION (CLAPPING PEAK:ORBBEC END TIME)\n----------------------------\n')

  # visualize MA data before synchronization point
  x = np.linspace(0,int(len(ma_synch)),int(len(ma_synch)))     # create timeframe using motion data length
  plt.plot(x,ma_synch['L.WristY'])        # plot the motion joint of interest
  plt.title(f'MA Clapping Motion WristLeftY - BEFORE SYNCHRONIZATION')
  plt.xlabel('Frames')
  plt.ylabel('Distance [cm]')
  plt.show()

  # zoom in on the peak movements
  # height = int(input("Minimal Peak Threshold: "))
  # distance = int(input("Minimal Distance Between Peaks (>1): "))
  height = ma_thresh
  distance = ma_dist
  x = ma_synch['L.WristY']
  peaks, _ = find_peaks(x, height= height, distance = distance)
  plt.plot(x)
  plt.plot(peaks, x[peaks], "x")
  plt.plot(np.zeros_like(x), "--", color="gray")
  plt.show()

  # show the peaks and corresponding index
  print('\nPeaks and Corresponding Index:') 
  print(x[peaks])

  # locate the peak of the third clap
  # ma_peak = int(input('\nIndex for peak of 3rd clap: ')) 
  ma_peak = ma_peak

  # overall MA game duration to locate end frame of the game 
  duration = ma_synch.Time.iloc[ma_peak] + (op_synch.Time.iloc[-1] - op_synch.Time.iloc[0])
  # round to two decimal places as MA time increases in .05 second decimal increments
  duration = round(duration,2)
  # locate the end frame of the game
  end_frame_ma = ma_synch[ma_synch.Time == duration].index[0]
  print(f'Motion Ending Frame (FROM ORBBEC): {end_frame_ma}')

  # cut motion data from starting position (horizontal peak) to frame at end of game log
  print(f'\nShape of Motion BEFORE synchronization: {ma_synch.shape}')
  ma_synch = ma_synch[ma_peak:end_frame_ma]           
  print(f'Shape of Motion AFTER synchronization: {ma_synch.shape} \n')

  # visualize MA data after synchronization point
  x = np.linspace(0,int(len(ma_synch)),int(len(ma_synch)))     # create timeframe using motion data length
  plt.plot(x,ma_synch['L.WristY'])        # plot the motion joint of interest
  plt.title(f'MA Clapping Motion L.WristY - AFTER Synchronization')
  plt.xlabel('Frames')
  plt.ylabel('Distance [cm]')
  plt.show()

  return ma_synch


def resam_ma(ma_resam, op_synch):
  '3.2.5) Resample (30 Hz)'

  print('\n3.2.5) RESAMPLE (30 HZ)\n----------------------------\n')

  print(f'The total time that motion runs is {ma_resam.Time.iloc[-1] - ma_resam.Time.iloc[0]} secs \n')
  print(f'Shape of Motion BEFORE resampling (downsampling to 30 Hz): {ma_resam.shape}')

  # save column and row to recreate dataframe
  ma_col = ma_resam.columns
  ma_row = ma_resam.index

  # resample motion dataframe
  secs = (ma_resam.Time.iloc[-1] - ma_resam.Time.iloc[0])
  samps = op_synch.shape[0] #int(secs*30)                # number of samples to resample
  ma_resam = signal.resample(ma_resam, samps)            # array of resampled data
  ma_resam = pd.DataFrame(ma_resam, columns = ma_col)    # recreate the dataframe
  print(f'Shape of Motion AFTER resampling (downsampling to 30 Hz): {ma_resam.shape} \n')

  # check which columns have null values from the resampled data 
  print('Columns with null values AFTER RESAMPLING:')
  for col in ma_resam.columns:
    null = str(ma_resam[col].isnull().unique()[0])
    if null == 'True':
      print('',col)  

  # visualize MA data after resampling
  x = np.linspace(0,int(len(ma_resam)),int(len(ma_resam)))     # create timeframe using motion data length
  plt.plot(x,ma_resam['L.WristY'])        # plot the motion joint of interest
  plt.title(f'Resampled MA Clapping Motion L.WristY')
  plt.xlabel('Frames')
  plt.ylabel('Distance [cm]')
  plt.show()

  return ma_resam


def track_ma(ma_track, untracked_op_index):
  'REMOVE UNTRACKED MA'

  print('\nREMOVE UNTRACKED MA\n----------------------------\n')

  # only keep the rows with TRACKED values from important columns
  print(f'shape of data BEFORE REMOVING UNTRACKED:{ma_track.shape}')
  ma_track = ma_track.drop(ma_track.index[untracked_op_index])
  print(f'shape of data AFTER REMOVING UNTRACKED:{ma_track.shape}\n') 

  return ma_track


def filte_ma(ma_filte, ma_hz):
  '3.2.6) Filter Noise (Cortex Software)'

  print('\n3.2.6) FILTER NOISE (CORTEX SOFTWARE)\n----------------------------\n')

  # save column names to rename new filtered data
  col_names = list(ma_filte.columns)              
  time = ma_filte['Time'].reset_index()
  del time['index']

  empty = []                                 
  # filter all but the first time column and store filtered array data in a list
  for col in ma_filte.columns[1:]:                  
    y = ma_filte[col]                          # for each column data
    y2 = butter_lowpass_filter(y,6,ma_hz,4)    # filter the column data
    empty.append(y2)                           # store filtered data

  # create dataframe from stored filtered data & rename columns & reinsert time column
  ma_filte = pd.DataFrame(empty).T                   
  ma_filte.columns = col_names[1:]                 
  ma_filte.insert(0,'Time',time)    

  return ma_filte                           


def cut_data(op_cut, ma_cut):
  
  if len(op_cut) > len(ma_cut):
    print(f'[Raw Length --> OP:{len(op_cut)} > MA:{len(ma_cut)}]')
    op_cut = op_cut.iloc[50:len(ma_cut)]
    ma_cut = ma_cut.iloc[50:]
    print(f'\n[Cut Length --> OP:{len(op_cut)} = MA:{len(ma_cut)}]')

  elif len(ma_cut) > len(op_cut):
    print(f'[Raw Length --> MA:{len(ma_cut)} > OP:{len(op_cut)}]')
    ma_cut = ma_cut.iloc[50:len(op_cut)]
    op_cut = op_cut.iloc[50:]
    print(f'\n[Raw Length --> OP:{len(op_cut)} = MA:{len(ma_cut)}]')

  else:
    print(f'[Raw Length --> MA:{len(ma_cut)} == OP:{len(op_cut)}]')
    ma_cut = ma_cut.iloc[50:]
    op_cut = op_cut.iloc[50:]
    print(f'\n[Raw Length --> OP:{len(op_cut)} = MA:{len(ma_cut)}]')

  return op_cut, ma_cut


def compare_coordinates(op_align, ma_align, beforevsafter):
  # Compare coordinates for OP vs MA - BEFORE ALIGN
  op_x = op_align['ShoulderLeftX'].mean()
  op_y = op_align['ShoulderLeftY'].mean()
  op_z = op_align['ShoulderLeftZ'].mean()
  marker_x = ma_align['OPX'].mean()
  marker_y = ma_align['OPY'].mean()
  marker_z = ma_align['OPZ'].mean()
  ma_x = ma_align['L.ShoulderX'].mean()
  ma_y = ma_align['L.ShoulderY'].mean()
  ma_z = ma_align['L.ShoulderZ'].mean()
  print(f'Average Coordinate Location of OP vs MA (LEFT SHOULDER): {beforevsafter} ALIGN\n')
  print(f'OP Left Shoulder X: \t{round(op_x,3)} cm \nMarker Depth Coordinate X: \t{round(marker_x,3)} cm \nMA Left Shoulder X: \t{round(ma_x,3)} cm \n')
  print(f'OP Left Shoulder Y: \t{round(op_y,3)} cm \nMarker Lateral Coordinate Y: \t{round(marker_y,3)} cm \nMA Left Shoulder Y: \t{round(ma_y,3)} cm\n')
  print(f'OP Left Shoulder Z: \t{round(op_z,3)} cm \nMarker Height Coordinate Z: \t{round(marker_z,3)} cm \nMA Left Shoulder Z: \t{round(ma_z,3)} cm \n')


def align_joints(op_align_joints, ma_align_joints):
  # Compare coordinates for OP vs MA - BEFORE ALIGN
  #compare_coordinates(op_align_joints, ma_align_joints, beforevsafter = 'BEFORE')

  # Realign All Synchronized Joint Data (39 graphs total)
  op_head = ['Head']
  ma_head = ['Front.Head']
  op_joints = ['Shoulder','Elbow','Wrist','Hip','Knee','Foot']
  ma_joints = ['Shoulder','Elbow','Wrist','ASIS','Knee','Ankle']
  op_side = ['Left','Right']
  ma_side = ['L.','R.']
  xyz = ['Y','Z','X']
  # Head Data
  for i in range(len(op_head)):
    for k in range(len(xyz)):
      op_joint = op_head[i] + xyz[k]
      ma_joint = ma_head[i] + xyz[k]
      joint = ma_joint
      if xyz[k] == 'Y':
        align_y = (ma_align_joints[ma_joint] - op_align_joints[op_joint])   
        align_y = remove_outliers(align_y)
        align_y = align_y.mean()                                           # alignment value in Y
        ma_align_joints[ma_joint] = ma_align_joints[ma_joint] - align_y 
      elif xyz[k] == 'Z':
        align_z = (op_align_joints[op_joint] - ma_align_joints[ma_joint])    
        align_z = remove_outliers(align_z)
        align_z = align_z.mean()                                           # alignment value in Z  
        op_align_joints[op_joint] = op_align_joints[op_joint] - align_z 
      elif xyz[k] == 'X':
        align_x = (ma_align_joints[ma_joint] - op_align_joints[op_joint])    
        align_x = remove_outliers(align_x)
        align_x = align_x.mean()                                           # alignment value in X
        ma_align_joints[ma_joint] = ma_align_joints[ma_joint] - align_x
  # Body Data
  for i in range(len(op_joints)):                   # for each joints
    for j in range(len(op_side)):                   # for each sides 
      for k in range(len(xyz)):                     # for each xyz 
        op_joint = op_joints[i] + op_side[j] + xyz[k]  # specific OP joint name
        ma_joint = ma_side[j] + ma_joints[i] + xyz[k]  # specific MA joint name 
        joint = ma_side[j] + ma_joints[i] + xyz[k]     # joint of interest
        if xyz[k] == 'Y':
          align_y = ma_align_joints[ma_joint] - op_align_joints[op_joint]   
          align_y = remove_outliers(align_y)
          align_y = align_y.mean()                                           # alignment value in Y
          ma_align_joints[ma_joint] = ma_align_joints[ma_joint] - align_y 
        elif xyz[k] == 'Z':
          align_z = (op_align_joints[op_joint] - ma_align_joints[ma_joint])    
          align_z = remove_outliers(align_z)
          align_z = align_z.mean()                                           # alignment value in Z  
          op_align_joints[op_joint] = op_align_joints[op_joint] - align_z 
        elif xyz[k] == 'X':
          align_x = (ma_align_joints[ma_joint] - op_align_joints[op_joint])    
          align_x = remove_outliers(align_x)
          align_x = align_x.mean()                                           # alignment value in X
          ma_align_joints[ma_joint] = ma_align_joints[ma_joint] - align_x

  # Compare coordinates for OP vs MA - AFTER ALIGN
  #compare_coordinates(op_align_joints, ma_align_joints, beforevsafter = 'AFTER')

  return op_align_joints, ma_align_joints


def data_vis(op_vis, ma_vis, joint, op_joint, ma_joint):
  '''
  To Visualize and Compare Column Data
  '''
  
  # create timeframe using shorter data length
  if len(op_vis) > len(ma_vis):
    print('[length OP > MA]')
    x = np.linspace(0,len(ma_vis),len(ma_vis))                       
    plt.plot(x,op_vis[op_joint].iloc[0:len(ma_vis)], label = 'OP Cleaned')                  
    plt.plot(x,ma_vis[ma_joint], label = 'MA Cleaned')                    
    plt.legend()
    plt.title(f'Cleaned Orbbec & Motion Data {joint}')
    plt.xlabel('Frames [Hz]')
    plt.ylabel('Distance [cm]')
    plt.show()
  elif len(ma_vis) > len(op_vis):     
    print('[length MA > OP]')
    x = np.linspace(0,len(op_vis),len(op_vis))                       
    plt.plot(x,op_vis[op_joint], label = 'OP Cleaned')                  
    plt.plot(x,ma_vis[ma_joint].iloc[0:len(op_vis)], label = 'MA Cleaned')                    
    plt.legend()
    plt.title(f'Cleaned Orbbec & Motion Data {joint}')
    plt.xlabel('Frames [Hz]')
    plt.ylabel('Distance [cm]')
    plt.show()
  else: 
    print('[length OP = MA: Data has already been CUT]')
    x = np.linspace(0,len(op_vis),len(op_vis))                       
    plt.plot(x,op_vis[op_joint], label = 'OP Cleaned')                  
    plt.plot(x,ma_vis[ma_joint], label = 'MA Cleaned')                    
    plt.legend()
    plt.title(f'Cleaned Orbbec & Motion Data {joint}')
    plt.xlabel('Frames [Hz]')
    plt.ylabel('Distance [cm]')
    plt.show()