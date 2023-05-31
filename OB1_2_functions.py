'''
---
# **OB1 AutoAnalysis 1-2) Body Segment Length**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

Soowan Choi
'''


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
    plt.figure(figsize=(5,3))                     
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
    plt.figure(figsize=(5,3))                     
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
    plt.figure(figsize=(5,3))                    
    plt.plot(x,op_vis[op_joint], label = 'OP Cleaned')                  
    plt.plot(x,ma_vis[ma_joint], label = 'MA Cleaned')                    
    plt.legend()
    plt.title(f'Cleaned Orbbec & Motion Data {joint}')
    plt.xlabel('Frames [Hz]')
    plt.ylabel('Distance [cm]')
    plt.show()


def fivetwo(op_final, ma_final, participant):
  '''
  To Calculate the CV of Each Body Segment (Upper Arm, Arm, Thigh, Shank, Trunk)
  '''

  # calculate the body segment length: sqrt( (Prox_x - Dist_x)^2 + (Prox_y - Dist_y)^2 + (Prox_z - Dist_z)^2 )

  # OP Body Segments: Proximal - Distal (LEFT EXAMPLE)
  # Upper Arm = 'ShoulderLeft_' - 'ElbowLeft_' 
  # Arm = 'ElbowLeft_' - 'WristLeft_'
  # Thigh = 'HipLeft_' - 'KneeLeft_'
  # Shank = 'KneeLeft_' - 'AnkleLeft_'
  # Trunk = 'SpineShoulder_' - 'SpineBase_'

  # MA Body Segments: Proximal - Distal (LEFT EXAMPLE)
  # Upper Arm = 'L.Shoulder_' - 'L.Elbow_'
  # Arm = 'L.Elbow_' - 'L.Wrist_'
  # Thigh = 'L.ASIS_' - L.Knee_'
  # Shank = 'L.Knee_' - 'L.Ankle_'
  # Trunk = (('R.Shoulder_' - 'L.Shoulder_')/2 + 'L.Shoulder_') - (('R.ASIS_' - 'L.ASIS_')/2 + 'L.ASIS_')   ***use if statement to check for smaller value



  # create list for body segments
  # All Joint Data (36 total)
  seg = ['Left Upper Arm', 'Right Upper Arm', 'Left Arm', 'Right Arm', 'Left Thigh', 'Right Thigh', 'Left Shank', 'Right Shank']

  op_joints = ['Shoulder','Elbow','Wrist','Hip','Knee','Foot']
  ma_joints = ['Shoulder','Elbow','Wrist','ASIS','Knee','Ankle']
  op_side = ['Left','Right']
  ma_side = ['L.','R.']
  xyz = ['X','Y','Z']

  op_ = []
  ma_ = []

  for i in range(len(op_joints)):                      # for each joints
    for j in range(len(op_side)):                      # for each sides 
      for k in range(len(xyz)):                        # for each xyz 
        op_joint = op_joints[i] + op_side[j] + xyz[k]  # specific OP joint name
        ma_joint = ma_side[j] + ma_joints[i] + xyz[k]
        op_.append(op_joint)
        ma_.append(ma_joint)



  # OP - Upper Arm Left, Upper Arm Right, Arm Left, Arm Right, Thigh Left, Thigh Right, Shank Left, Shank Right
  j = 0
  col_seg = []
  cov_seg = []
  for i in range(0,len(op_)-6,3):
    # to avoid 'Wrist' - 'Hip' segment
    if 'Wrist' in op_[i]:
      continue
    # calculate the body segment length
    op_x = op_final[op_[i]] - op_final[op_[i+6]]          # order of subtraction does not matter since negative values will be squared!
    op_y = op_final[op_[i+1]] - op_final[op_[i+7]]
    op_z = op_final[op_[i+2]] - op_final[op_[i+8]]
    op_seg = np.sqrt(op_x**2 + op_y**2 + op_z**2)
    # calculate the coefficient of variation
    op_cv = op_seg.std() / op_seg.mean()                  # CV = std / mean
    col_seg.append(seg[j] + '[%]')
    cov = round(op_cv*100, 3)
    cov_seg.append(cov)
    print(f'OP - CV of {seg[j]}: {op_cv * 100}')          # coefficient of variation as a percentage
    j += 1
    print(f'mean: {op_seg.mean()}, min: {op_seg.min()}, max: {op_seg.max()} \n')

  # OP - Trunk
  op_x = op_final['SpineShoulderX'] - op_final['SpineBaseX']  # order of subtraction does not matter since negative values will be squared!
  op_y = op_final['SpineShoulderY'] - op_final['SpineBaseY']
  op_z = op_final['SpineShoulderZ'] - op_final['SpineBaseZ']
  op_seg = np.sqrt(op_x**2 + op_y**2 + op_z**2)
  # calculate the coefficient of variation
  op_cv = op_seg.std() / op_seg.mean()                        # CV = std / mean
  col_seg.append('Trunk' + '[%]')
  cov = round(op_cv*100, 3)
  cov_seg.append(cov)
  print(f'OP - CV of Trunk: {op_cv * 100}')                   # coefficient of variation as a percentage
  print(f'mean: {op_seg.mean()}, min: {op_seg.min()}, max: {op_seg.max()} \n')

  op_cov = pd.DataFrame(np.reshape(cov_seg, (1, 9)), columns = col_seg, index = participant)



  # MA - Upper Arm Left, Upper Arm Right, Arm Left, Arm Right, Thigh Left, Thigh Right, Shank Left, Shank Right
  j = 0
  for i in range(0,len(ma_)-6,3):
    # to avoid 'Wrist' - 'Hip' segment
    if 'Wrist' in op_[i]:
      continue
    # calculate the body segment length
    ma_x = ma_final[ma_[i]] - ma_final[ma_[i+6]]          # order of subtraction does not matter since negative values will be squared!
    ma_y = ma_final[ma_[i+1]] - ma_final[ma_[i+7]]
    ma_z = ma_final[ma_[i+2]] - ma_final[ma_[i+8]]
    ma_seg = np.sqrt(ma_x**2 + ma_y**2 + ma_z**2)
    # calculate the coefficient of variation
    ma_cv = ma_seg.std() / ma_seg.mean()                  # CV = std / mean
    print(f'MA - CV of {seg[j]}: {ma_cv * 100}')          # coefficient of variation as a percentage
    j += 1
    print(f'mean: {ma_seg.mean()}, min: {ma_seg.min()}, max: {ma_seg.max()} \n')

  # MA - Trunk
  ma_x = ((ma_final['R.ShoulderX'] + ma_final['L.ShoulderX']) / 2) - ((ma_final['R.ASISX'] + ma_final['L.ASISX']) / 2) # order of subtraction does not matter since negative values will be squared!
  ma_y = ((ma_final['R.ShoulderY'] + ma_final['L.ShoulderY']) / 2) - ((ma_final['R.ASISY'] + ma_final['L.ASISY']) / 2)
  ma_z = ((ma_final['R.ShoulderZ'] + ma_final['L.ShoulderZ']) / 2) - ((ma_final['R.ASISZ'] + ma_final['L.ASISZ']) / 2)
  ma_seg = np.sqrt(ma_x**2 + ma_y**2 + ma_z**2)
  # calculate the coefficient of variation
  ma_cv = ma_seg.std() / ma_seg.mean()               # CV = std / mean
  print(f'OP - CV of Trunk: {ma_cv * 100}')          # coefficient of variation as a percentage
  print(f'mean: {ma_seg.mean()}, min: {ma_seg.min()}, max: {ma_seg.max()} \n')

  return op_cov