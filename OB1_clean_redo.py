# Method 3: Method 2 Fixed
# (OP: XYZ MA: XYZ -->  Post0316 MA: XZnegY Pre0316 MA: YZX)

'''
---
# **Copy of CHAPTER 3: OB2 Clean Raw Auto**
---

Soowan Choi
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# from OB2_clean_raw_auto_fun import * # todo import other modules
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



def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y



def filter(data_filte, data_hz, cutoff):
  # save column names to rename new filtered data
  col_names = list(data_filte.columns)              
  time = data_filte['Time'].reset_index()
  del time['index']

  empty = []                                 
  # filter all but the first time column and store filtered array data in a list
  for col in data_filte.columns[1:]:                  
    y = data_filte[col]                          # for each column data
    y2 = butter_lowpass_filter(y, cutoff, data_hz, 6)    # filter the column data (data, 1-3-6-9-12 Hz cutoff, fs, 6th order)
    empty.append(y2)                             # store filtered data

  # create dataframe from stored filtered data & rename columns & reinsert time column
  data_filte = pd.DataFrame(empty).T                   
  data_filte.columns = col_names[1:]                 
  data_filte.insert(0,'Time',time)    

  return data_filte    



def data_vis(op_vis, ma_vis, joint):
  '''
  To Visualize and Compare Column Data
  '''

   # If resampled so OP and MA have equal data length
  if len(op_vis) == len(ma_vis):
    print('[length OP == MA]')
    x = np.linspace(0,len(ma_vis),len(ma_vis))  
    plt.figure(figsize=(5,3))                     
    plt.plot(x,op_vis, label = 'OP Cleaned')                  
    plt.plot(x,ma_vis, label = 'MA Cleaned')                    
    plt.legend()
    plt.title(f'Cleaned Orbbec & Motion Data {joint}')
    plt.xlabel('Frames [Hz]')
    plt.ylabel('Distance [cm]')
    plt.show()
  
  # If not resampled so OP and MA have different data length
  else:
    x = np.linspace(0,len(op_vis),len(op_vis))  
    plt.figure(figsize=(5,3))                     
    plt.plot(x,op_vis, label = 'OP Cleaned')
    plt.legend()
    plt.title(f'Cleaned Orbbec Data {joint}')
    plt.xlabel('Frames [Hz]')
    plt.ylabel('Distance [cm]')
    plt.show()   
    
    x = np.linspace(0,len(ma_vis),len(ma_vis))  
    plt.figure(figsize=(5,3))                
    plt.plot(x,ma_vis, label = 'MA Cleaned')                    
    plt.legend()
    plt.title(f'Cleaned Motion Data {joint}')
    plt.xlabel('Frames [Hz]')
    plt.ylabel('Distance [cm]')
    plt.show()



def align_vis(offset_bias, joint, coord):
  '''
  To Visualize and Compare Column Data
  '''
  x = np.linspace(0,len(offset_bias),len(offset_bias))  
  plt.figure(figsize=(5,3))                     
  plt.plot(x, offset_bias, label = 'Offset_Bias')                                  
  plt.legend()
  plt.title(f'Offset_Bias OP & MA: {joint} {coord}')
  plt.xlabel('Frames [Hz]')
  plt.ylabel('Distance [cm]')
  plt.show()



def resam_ma(ma_resam, op_synch):
  # to resampled data
  from scipy import signal
  from sklearn.utils import resample
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

  return ma_resam



def nullv_ma(ma_nullv):
  '3.2.3) INTERPOLATE NULL'

  print('\n3.2.3) INTERPOLATE NULL\n----------------------------\n')

  # columns with NULL values BEFORE INTERPOLATING 
  nulls = []
  print('Joints with NULL values BEFORE INTERPOLATING NULL:')
  for col in ma_nullv.columns:
    null = ma_nullv[col].isnull().unique()
    if len(null) > 1 or True in null:
      nulls.append(col)
  print(nulls)

  # Important columns with NULL values BEFORE INTERPOLATING
  isna = []
  important = ['Front.Head','.Shoulder','.Elbow','.Wrist','.ASIS','.Knee','.Ankle']
  print('\nIMPORTANT Joints with NULL values BEFORE INTERPOLATING NULL:')
  for i in range(len(nulls)):
    for joint in important:
      if joint in nulls[i]:
        isna.append(nulls[i])
        isna.append(nulls[i+1])
        isna.append(nulls[i+2])
  print(isna)
  print('\n')

  # count how many NULL values to clean in each important column
  for col in isna:
    if "Unnamed" not in col:
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
    if "Unnamed" not in col:
      for index, val in enumerate(ma_nullv[col]):
        if str(val) == 'nan':
          null_index.append(index)

  # check if null appears at middle of data (instead of beginning or end)
  for i in range(len(null_index) - 1):
    if (null_index[i+1] - null_index[i]) != 1:
      print(f"Null Appears Middle of Data: {null_index[i-3:i+3]}")

  # interpolate and fill null values 
  for col in isna:    
    ma_nullv[col] = ma_nullv[col].astype(float)
    ma_nullv[col] = ma_nullv[col].interpolate(method = 'polynomial', order = 2, limit_direction = 'forward')
    ma_nullv[col] = ma_nullv[col].astype(str)
  
  # ***backwards fill AFTER initial interpolation, this data will be REMOVED later but required now as placeholders***
  for col in isna:    
    ma_nullv[col] = ma_nullv[col].astype(float)
    ma_nullv[col] = ma_nullv[col].interpolate(method = 'linear', limit_direction = 'backward')
    ma_nullv[col] = ma_nullv[col].astype(str)
  # Interpolate remaining missing values near end of dataframe
  for col in isna:    
    ma_nullv[col] = ma_nullv[col].astype(float)
    ma_nullv[col] = ma_nullv[col].interpolate(method = 'linear')
    ma_nullv[col] = ma_nullv[col].astype(str)

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
  important = ['.Shoulder','.Elbow','.Wrist','.ASIS','.Knee','.Ankle']
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
        # Drop null values at end of frames
        ma_nullv = ma_nullv.drop(ma_nullv.index[i])
        count += 1
    print(f"{count} NULL Values in the {col} column")



  return ma_nullv



def coord_op(op_coord):
  '2.2.2) Coordinate Transformation (Y ← OP_X | Z ← OP_Y | X ← OP_Z) + (Y ← MA_X | Z ← MA_Z | X ← negMA_Y)'

  print('\n2.2.2) COORDINATE TRANSFORMATION (Y ← OP_X | Z ← OP_Y | X ← OP_Z) + (Y ← MA_X | Z ← MA_Z | X ← negMA_Y)\n----------------------------\n')

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



def coord_ma(ma_coord, pre_or_post):
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
  if pre_or_post == "pre0316":
     # We changed OP to MA
     naming_system = ['X','Y','Z']  
  elif pre_or_post == "post0316":
     # We changed both OP and MA separately
     naming_system = ['Y','X','Z']    
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





def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 

  return op 

def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file, header = 3) 
  
  return ma





# CREATE FILES (DATE & GAMES)
# ***0404_P10 MA File Named Single1 and Single2 vs Single and Single***
# Files with Problems: 0408_P18_BC8

# op_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
# ma_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
op_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
            'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
ma_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
            'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']

# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']

# Specify if Repetition Count exercise or Repetition Timer exercise
count = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
            'StarJump', 'SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
timer = ['SeatClfStr', 'Run', 'ForStep', 'CalfStr', 'TdemStnce']


directory_unknown = []
game_peaks_unknown = []
# Automatically Loop Through Each Participant
for mmdd_p in mmdd_p_all:
  # Automatically Loop Through Each Game
  for game_ind in range(len(op_games)):
    print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
    op_file = '2023' + mmdd_p[:4] + '-' + op_games[game_ind] + "-Data.csv"
    ma_file = '2023' + mmdd_p[:4] + '-' + ma_games[game_ind] + ".csv"


    try:
      # Load OP Data
      op = load_op('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file)

      # Load MA Data
      ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/MA_' + mmdd_p + '/' + ma_file)

    except FileNotFoundError:
      # if directory game file doesn't exist, go to next game
      directory_unknown.append(op_file)
      continue


    # Load automatic peak values to clean
    if mmdd_p in ['0221_P01', '0314_P02', '0314_P03', '0315_P04']:
       peaks = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/peaks_pre0316.csv") 
    else:
       peaks = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/peaks_post0316.csv") 

    for i in range(len(peaks)):
        if peaks["Date_P##"][i] == mmdd_p:
          if str(peaks[op_games[game_ind]][i]) == 'nan':
              # if we dont know the peaks, go to next game
              game_peaks_unknown.append(mmdd_p + "_" + op_games[game_ind])
              break
          else:
              # if we know the peaks, clean each sensor for the game
              # print(peaks[op_games[game_ind]][i:i+7])
              op_thresh = int(peaks[op_games[game_ind]][i])
              op_dist = int(peaks[op_games[game_ind]][i+1])
              op_peak = int(peaks[op_games[game_ind]][i+2])
              op_end = int(peaks[op_games[game_ind]][i+3])
              ma_thresh = int(peaks[op_games[game_ind]][i+4])
              ma_dist = int(peaks[op_games[game_ind]][i+5])
              ma_peak = int(peaks[op_games[game_ind]][i+6])


              # Clean OP 
              # Synchronize
              op_synch = op.drop(op.index[0:op_peak])
              # Standardize Time 
              op_synch['Time'] = op_synch['Time'] - op_synch.iloc[0,0]
              op_final = op_synch
     
              # Clean MA
              # Synchronize
              ma_synch = ma.drop(ma.index[:ma_peak])
              # Standardize Time 
              ma_synch['Time'] = ma_synch['Time'] - ma_synch.iloc[0,1]
              # Standardize Frames
              ma_synch['Frame#'] = ma_synch['Frame#'] - (ma_synch.iloc[0,0] - 1)
              ma_synch['Frame#'] = ma_synch['Frame#'].astype(int)
              ma_final = ma_synch

              # Synchronize End of MA too
              if (op_final.Time.iloc[-1]/1000) < (ma_final.Time.iloc[-1]):
                # overall OP game duration to locate end frame of the game for MA 
                duration = (op_final.Time.iloc[-1] - op_final.Time.iloc[0]) / 1000
                # round to two decimal places as MA time increases in .05 second decimal increments
                duration = round(duration,2)
                # locate the end frame of the game
                end_frame_ma = ma_final.loc[ma_final['Time'] >= duration, 'Frame#'].values[0]
                ma_final = ma_final.drop(ma_final.index[end_frame_ma:])
              # Or OP if OP games runs longer
              else:
                # overall MA game duration to locate end frame of the game for OP 
                duration = (ma_final.Time.iloc[-1] - ma_final.Time.iloc[0]) 
                # round to two decimal places as MA time increases in .05 second decimal increments
                duration = round(duration,2)
                # locate the end frame of the game
                op_index = op_final.reset_index()
                end_frame_op = op_index[op_index.Time/1000 >= duration].index[0]
                op_final = op_final.drop(op_final.index[end_frame_op:])
                 


              # Interpolate Missing MA Values
              ma_final = nullv_ma(ma_final)

              # Resample MA to Match OP --> Used to Find Offset_Bias
              ma_resam = resam_ma(ma_final, op_final)





              # Align the Joints (OP: XYZ MA: XYZ -->  Post0316 MA: XZnegY Pre0316 MA: YZX)
              op_joints = ['Head','ShoulderRight','ElbowRight','WristRight','ShoulderLeft','ElbowLeft','WristLeft',
                           'HipRight','HipLeft','KneeRight','FootRight','KneeLeft','FootLeft']
              ma_joints = ['Front.Head','R.Shoulder','R.Elbow','R.Wrist','L.Shoulder','L.Elbow','L.Wrist',
                           'R.ASIS','L.ASIS','R.Knee','R.Ankle','L.Knee','L.Ankle']
              
              ma_joint_cols = []
              op_joint_cols = []
              # For each joint
              for i in range(len(op_joints)):
                # For each of the columns in each file
                for n in range(len(ma_resam.columns) - 2):
                   # If correct joint column
                   if ma_joints[i] == ma_resam.columns[n]:
                      # Check to make sure correct joint columns 
                      ma_joint_cols.append(ma_resam.columns[n])
                      ma_joint_x = ma_resam.iloc[:,n]
                      ma_joint_y = ma_resam.iloc[:,n+1]
                      ma_joint_z = ma_resam.iloc[:,n+2]
                for k in range(1, len(op_final.columns)-2, 3):
                   if (op_joints[i] in op_final.columns[k]) and ('Tracked' not in op_final.columns[k]):
                      # print(op_joints[i], op_final.columns[k])
                      # Check to make sure correct joint columns 
                      print(op_final.columns[k])
                      op_joint_cols.append(op_final.columns[k-2])
                      op_joint_x = op_final.iloc[:,k]
                      op_joint_y = op_final.iloc[:,k+1]
                      op_joint_z = op_final.iloc[:,k+2]

                # Convert values to float (MA used to be 'str')
                # MA mm --> cm and OP m --> cm  
                #print(f'MA: {type(ma_joint_x.iloc[-100])}, OP: {type(op_joint_x.iloc[-100])}')
                ma_joint_x = ma_joint_x.astype(float) / 10
                ma_joint_y = ma_joint_y.astype(float) / 10
                ma_joint_z = ma_joint_z.astype(float) / 10
                op_joint_x = op_joint_x.astype(float) * 100
                op_joint_y = op_joint_y.astype(float) * 100
                op_joint_z = op_joint_z.astype(float) * 100
                #print(f'MA: {type(ma_joint_x.iloc[-1])}, OP: {type(op_joint_x.iloc[-1])}')

                # Get mean coordinate offset (note different data length of MA and OP)(note different coordinate mapping Post0316)
                # Post0316 (MA_X - OP_X | MA_Y - OP_Z | MA_Z - OP_Y)
                if mmdd_p[-3:] != 'P01' and mmdd_p[-3:] != 'P02' and mmdd_p[-3:] != 'P03' and mmdd_p[-3:] != 'P04':
                  offset_bias_x = ma_joint_x - op_joint_x
                  offset_bias_y = -1*ma_joint_y - op_joint_z
                  offset_bias_z = ma_joint_z - op_joint_y
                  offset_bias_x = remove_outliers(offset_bias_x)
                  offset_bias_y = remove_outliers(offset_bias_y)
                  offset_bias_z = remove_outliers(offset_bias_z)
                  # align_vis(offset_bias_x, op_joints[i], 'X')
                  # align_vis(offset_bias_y, op_joints[i], 'Y')
                  # align_vis(offset_bias_z, op_joints[i], 'Z')
                  offset_bias_x = offset_bias_x.mean()
                  offset_bias_y = offset_bias_y.mean()
                  offset_bias_z = offset_bias_z.mean()
                  # data_vis(op_joint_x, ma_joint_x - offset_bias_x, op_joints[i] + 'X')
                  # data_vis(op_joint_z, -1*ma_joint_y - offset_bias_y, op_joints[i] + 'Y')
                  # data_vis(op_joint_y, ma_joint_z - offset_bias_z, op_joints[i] + 'Z')
                  # print(f'{op_joints[i]} X: \t Offset_Bias: {round(offset_bias_x,3)}cm \tOP: {round(op_joint_x.mean(),3)}cm \tMA: {round(ma_joint_x.mean(),3)}cm')
                  # print(f'{op_joints[i]} Y: \t Offset_Bias: {round(offset_bias_y,3)}cm \tOP: {round(op_joint_z.mean(),3)}cm \tMA: {round((ma_joint_y).mean(),3)}cm')
                  # print(f'{op_joints[i]} Z: \t Offset_Bias: {round(offset_bias_z,3)}cm \tOP: {round(op_joint_y.mean(),3)}cm \tMA: {round(ma_joint_z.mean(),3)}cm')
                  # print(f'XYZ Offset and Bias in {op_joints[i]}: {offset_bias_x} {offset_bias_y} {offset_bias_z}')

                  # Subtract from MA file 
                  for m in range(len(ma_final.columns) - 2):
                      # If MA column isn't completely empty, align that joint column using that joint offset_bias
                      if str(offset_bias_x) != 'nan' and ma_joints[i] == ma_final.columns[m] and ma_joints[i] != "R.Shoulder":
                        # Align MA coordinates to match OP (remove systemic bias)
                        # MA cm --> mm
                        print("NAN?", str(offset_bias_x))
                        print("Joint:", ma_joints[i])
                        print('col:', ma_final.columns[m])
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(float) - offset_bias_x*10
                        ma_final.iloc[:,m+1] = (-1*(ma_final.iloc[:,m+1]).astype(float) - offset_bias_y*10)
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(float) - offset_bias_z*10
                        # Convert back to string datatype
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(str)
                        ma_final.iloc[:,m+1] = (ma_final.iloc[:,m+1]).astype(str)
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(str)
                        print('done col:', ma_final.columns[m])

                      # If R.Shoulder isn't empty, align R.Shoulder using R.Shoulder offset_bias
                      if str(offset_bias_x) != 'nan' and ma_joints[i] == ma_final.columns[m] and ma_joints[i] == "R.Shoulder":
                        print('col:', ma_final.columns[m])
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(float) - offset_bias_x*10
                        ma_final.iloc[:,m+1] = (-1*(ma_final.iloc[:,m+1]).astype(float) - offset_bias_y*10)
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(float) - offset_bias_z*10
                        # Convert back to string datatype
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(str)
                        ma_final.iloc[:,m+1] = (ma_final.iloc[:,m+1]).astype(str)
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(str)
                        print('done col:', ma_final.columns[m])
                        # If R.Offset isn't empty, align R.Offset using R.Shoulder offset_bias
                        if str(ma_final['R.Offset'].unique()[0]) != 'nan': 
                          print('col:', 'R.Offset')
                          ma_final.iloc[:,m+3] = (ma_final.iloc[:,m+3]).astype(float) - offset_bias_x*10
                          ma_final.iloc[:,m+4] = -1*(-1*(ma_final.iloc[:,m+4]).astype(float) - offset_bias_y*10)
                          ma_final.iloc[:,m+5] = (ma_final.iloc[:,m+5]).astype(float) - offset_bias_z*10
                          # Convert back to string datatype
                          ma_final.iloc[:,m+3] = (ma_final.iloc[:,m+3]).astype(str)
                          ma_final.iloc[:,m+4] = (ma_final.iloc[:,m+4]).astype(str)
                          ma_final.iloc[:,m+5] = (ma_final.iloc[:,m+5]).astype(str)
                          print('done col:', 'R.Offset')

              

                # Pre0316 (MA_X - OP_Z | MA_Y - OP_X | MA_Z - OP_Y)
                else: 
                  offset_bias_x = ma_joint_x - op_joint_z
                  offset_bias_y = ma_joint_y - op_joint_x
                  offset_bias_z = ma_joint_z - op_joint_y
                  offset_bias_x = remove_outliers(offset_bias_x)
                  offset_bias_y = remove_outliers(offset_bias_y)
                  offset_bias_z = remove_outliers(offset_bias_z)
                  # align_vis(offset_bias_x, op_joints[i], 'X')
                  # align_vis(offset_bias_y, op_joints[i], 'Y')
                  # align_vis(offset_bias_z, op_joints[i], 'Z')
                  offset_bias_x = offset_bias_x.mean()
                  offset_bias_y = offset_bias_y.mean()
                  offset_bias_z = offset_bias_z.mean()
                  # data_vis(op_joint_z, ma_joint_x - offset_bias_x, op_joints[i] + 'X')
                  # data_vis(op_joint_x, ma_joint_y - offset_bias_y, op_joints[i] + 'Y')
                  # data_vis(op_joint_y, ma_joint_z - offset_bias_z, op_joints[i] + 'Z')
                  # print(f'{op_joints[i]} X: \t Offset_Bias: {round(offset_bias_x,3)}cm \tOP: {round(op_joint_x.mean(),3)}cm \tMA: {round(ma_joint_x.mean(),3)}cm')
                  # print(f'{op_joints[i]} Y: \t Offset_Bias: {round(offset_bias_y,3)}cm \tOP: {round(op_joint_z.mean(),3)}cm \tMA: {round((ma_joint_y).mean(),3)}cm')
                  # print(f'{op_joints[i]} Z: \t Offset_Bias: {round(offset_bias_z,3)}cm \tOP: {round(op_joint_y.mean(),3)}cm \tMA: {round(ma_joint_z.mean(),3)}cm')
                  # print(f'XYZ Offset and Bias in {op_joints[i]}: {offset_bias_x} {offset_bias_y} {offset_bias_z}')

                  # Subtract from MA file 
                  for m in range(len(ma_final.columns) - 2):
                      # If MA column isn't completely empty, align that joint column using that joint offset_bias
                      if str(offset_bias_x) != 'nan' and ma_joints[i] == ma_final.columns[m] and ma_joints[i] != "R.Shoulder":
                        # Align MA coordinates to match OP (remove systemic bias)
                        # MA cm --> mm
                        print("NAN?", str(offset_bias_x))
                        print("Joint:", ma_joints[i])
                        print('col:', ma_final.columns[m])
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(float) - offset_bias_x*10
                        ma_final.iloc[:,m+1] = (ma_final.iloc[:,m+1]).astype(float) - offset_bias_y*10
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(float) - offset_bias_z*10
                        # Convert back to string datatype
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(str)
                        ma_final.iloc[:,m+1] = (ma_final.iloc[:,m+1]).astype(str)
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(str)
                        print('done col:', ma_final.columns[m])

                      # If R.Shoulder isn't empty, align R.Shoulder using R.Shoulder offset_bias
                      if str(offset_bias_x) != 'nan' and ma_joints[i] == ma_final.columns[m] and ma_joints[i] == "R.Shoulder":
                        print('col:', ma_final.columns[m])
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(float) - offset_bias_x*10
                        ma_final.iloc[:,m+1] = (ma_final.iloc[:,m+1]).astype(float) - offset_bias_y*10
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(float) - offset_bias_z*10
                        # Convert back to string datatype
                        ma_final.iloc[:,m] = (ma_final.iloc[:,m]).astype(str)
                        ma_final.iloc[:,m+1] = (ma_final.iloc[:,m+1]).astype(str)
                        ma_final.iloc[:,m+2] = (ma_final.iloc[:,m+2]).astype(str)
                        print('done col:', ma_final.columns[m])
                        # If R.Offset isn't empty, align R.Offset using R.Shoulder offset_bias
                        if str(ma_final['R.Offset'].unique()[0]) != 'nan': 
                          print('col:', 'R.Offset')
                          ma_final.iloc[:,m+3] = (ma_final.iloc[:,m+3]).astype(float) - offset_bias_x*10
                          ma_final.iloc[:,m+4] = (ma_final.iloc[:,m+4]).astype(float) - offset_bias_y*10
                          ma_final.iloc[:,m+5] = (ma_final.iloc[:,m+5]).astype(float) - offset_bias_z*10
                          # Convert back to string datatype
                          ma_final.iloc[:,m+3] = (ma_final.iloc[:,m+3]).astype(str)
                          ma_final.iloc[:,m+4] = (ma_final.iloc[:,m+4]).astype(str)
                          ma_final.iloc[:,m+5] = (ma_final.iloc[:,m+5]).astype(str)
                          print('done col:', 'R.Offset')
                  



              # Coordinate Transformation (Translation): Rename Column Names
              # Post0316
              if mmdd_p[-3:] != 'P01' and mmdd_p[-3:] != 'P02' and mmdd_p[-3:] != 'P03' and mmdd_p[-3:] != 'P04':
                 ma_final = coord_ma(ma_final, 'post0316')
              # Pre0316
              else:
                 ma_final = coord_ma(ma_final, 'pre0316')

              op_final = coord_op(op_final)


              # Interpolate Missing Values in MA
              ma_final = nullv_ma(ma_final)

              # Resample MA to Match OP --> Final Results
              ma_final = resam_ma(ma_final, op_final)
            
              # Convert OP (m --> cm) and MA values (mm --> cm) to cm
              op_final = op_final.astype(float)
              op_final.iloc[:,1:57] = op_final.iloc[:,1:57] *100
              ma_final = ma_final.astype(float)
              ma_final.iloc[:,2:] = ma_final.iloc[:,2:] / 10

              # Convert OP time values to seconds and set as MA time value
              op_final = op_final.reset_index().drop(columns=['index'])
              op_final.iloc[:,0] = op_final.iloc[:, 0] / 1000
              ma_final = ma_final.reset_index().drop(columns=['index'])
              ma_final.Time = op_final.iloc[:, 0] 

              # Filter OP ONLY (MA Filtered Using Cortex Software)
              # OP Frequency
              op_frames = len(op_synch.Time)
              op_seconds = (op_synch.Time.iloc[-1]/1000 - op_synch.Time.iloc[0]/1000)  
              op_hz = op_frames / op_seconds
              op_final = filter(op_final, op_hz, 3)  # Choose Cut-Off Frequency

              # # MA Frequency
              # ma_frames = len(ma_synch.Time)
              # ma_seconds = (ma_synch.Time.iloc[-1] - ma_synch.Time.iloc[0])  
              # ma_hz = ma_frames / ma_seconds
              # ma_final = ma_final.drop(columns=['Frame#'])
              # ma_final = filter(ma_final, ma_hz, 5)  # Choose Cut-Off Frequency
              
                 


              # # Visualize to check resampled data
              # x = np.linspace(0,len(op_final),len(op_final))  
              # plt.figure(figsize=(5,3))                     
              # plt.plot(x,op_final['WristLeftX'], label = 'OP Cleaned')                  
              # plt.plot(x,ma_final['L.WristX'], label = 'MA Cleaned')                    
              # plt.legend()
              # plt.title(f'Cleaned Orbbec & Motion Data L.WristX')
              # plt.xlabel('Frames [Hz]')
              # plt.ylabel('Distance [cm]')
              # plt.show()

              # x = np.linspace(0,len(op_final),len(op_final))  
              # plt.figure(figsize=(5,3))                     
              # plt.plot(x,op_final['WristLeftY'], label = 'OP Cleaned')                  
              # plt.plot(x,ma_final['L.WristY'], label = 'MA Cleaned')                    
              # plt.legend()
              # plt.title(f'Cleaned Orbbec & Motion Data L.WristY')
              # plt.xlabel('Frames [Hz]')
              # plt.ylabel('Distance [cm]')
              # plt.show()

              # x = np.linspace(0,len(op_final),len(op_final))  
              # plt.figure(figsize=(5,3))                     
              # plt.plot(x,op_final['WristLeftZ'], label = 'OP Cleaned')                  
              # plt.plot(x,ma_final['L.WristZ'], label = 'MA Cleaned')                    
              # plt.legend()
              # plt.title(f'Cleaned Orbbec & Motion Data L.WristZ')
              # plt.xlabel('Frames [Hz]')
              # plt.ylabel('Distance [cm]')
              # plt.show()





              # *** For each participant rename Power1 --> PowerR etc. ***
              if op_games[game_ind] == 'Power1' or op_games[game_ind] == 'Power2':
                  # Load power.csv file to rename
                  power = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/power.csv") 
                  # For each row 
                  # If correct participant
                  # For each column
                  # If correct Power column 
                  # Rename: BC#-Game
                  for i in range(len(power)):
                      if mmdd_p in power.iloc[i,0]:
                          for j in range(len(power.columns)):
                              if op_games[game_ind] == str(power.columns[j]):
                                  op_game = str(power.iloc[i,j])
                                  ma_game = str(power.iloc[i,j])
                                  break
                      

              # *** For each participant rename BC#-Game ***
              if 'BC' in op_games[game_ind]:
                # Load bootcamp.csv file to rename
                bootcamp = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/bootcamp.csv") 
                # For each row 
                # If correct participant
                # For each column
                # If correct BC column and corresponding cell isn't empty
                # Rename: BC#-Game
                for i in range(len(bootcamp)):
                    if mmdd_p in bootcamp.iloc[i,0]:
                        for j in range(len(bootcamp.columns)):
                            if op_games[game_ind] == str(bootcamp.columns[j]):
                                    if str(bootcamp.iloc[i,j]) != 'nan':
                                        op_game = op_games[game_ind] + '-' + str(bootcamp.iloc[i,j])
                                        ma_game = ma_games[game_ind] + '-' + str(bootcamp.iloc[i,j])
                                        break
                                    else:
                                        print(str(bootcamp.columns[j]),str(bootcamp.iloc[i,j]), op_games[game_ind])
                                        op_game = op_games[game_ind] + '-' + 'NA'
                                        ma_game = ma_games[game_ind] + '-' + 'NA'
                                        break


              # DOWNLOAD FILES TO DOWNLOADS FOLDER
              if 'Power' in op_games[game_ind] or 'BC' in op_games[game_ind]:
                # DOWNLOAD CLEANED OP DATA
                op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                # DOWNLOAD CLEANED MA BOOT CAMP DATA
                ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)
              else:
                # DOWNLOAD CLEANED OP DATA
                op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                # DOWNLOAD CLEANED MA BOOT CAMP DATA
                ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)
                 

              # # DOWNLOAD FILES TO SPECIFIC LOCATION
              # if 'Power' in op_games[game_ind] or 'BC' in op_games[game_ind]:
              #   # DOWNLOAD CLEANED OP DATA
              #   op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
              # else:
              #   # DOWNLOAD CLEANED OP DATA
              #   op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
                 

              # # DOWNLOAD FILES TO SUPER SPECIFIC LOCATION
              # if 'Power' in op_games[game_ind] or 'BC' in op_games[game_ind]:
              #   # DOWNLOAD CLEANED OP DATA
              #   op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/New_Clean_12Hz_20230627/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/New_Clean_12Hz_20230627/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
              # else:
              #   # DOWNLOAD CLEANED OP DATA
              #   op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/New_Clean_12Hz_20230627/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_0551/2023_{mmdd_p[:4]}_{mmdd_p[-3:]}/Auto_Clean_{mmdd_p[:4]}_{mmdd_p[-3:]}/New_Clean_12Hz_20230627/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
                 

              
print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)

print("\nFOLLOWING GAMES HAVE UNKNOWN PEAKS:", game_peaks_unknown)