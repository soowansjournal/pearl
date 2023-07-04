'''
---
# **OB1 AutoAnalysis 1-3) Joint Angle ROM**
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


def min_angle(op_angle, ma_angle, LR, joint, minmax):
  ''' to compare the average angle of BOTTOM 5% MIN Values'''

  # # Method 1 --> Avg of Lowest 5 Trough Values

  # # Filter
  # op_angle = butter_lowpass_filter(op_angle, 1, 30, 6)
  # ma_angle = butter_lowpass_filter(ma_angle, 1, 30, 6)
  # # Outliers
  # op_angle = remove_outliers(pd.Series(op_angle)).reset_index(drop = True)
  # ma_angle = remove_outliers(pd.Series(ma_angle)).reset_index(drop = True)
  # op_angle = np.array(op_angle)
  # ma_angle = np.array(ma_angle)

  # # locate the peaks and troughs from each OP array
  # # determine the horizontal distance between peaks
  # distance = len(op_angle) / 10
  # op_x = np.array(op_angle)
  # op_peaks, _= find_peaks(op_x, distance = distance)
  # op_troughs, _= find_peaks(-op_x, distance = distance)
  # plt.plot(op_x)
  # plt.plot(op_peaks,op_x[op_peaks], '^')
  # plt.plot(op_troughs,op_x[op_troughs], 'v')
  # plt.title(f'OP Peak Values')
  # plt.xlabel('Frames [Hz]')
  # plt.ylabel('Angle [deg]')
  # plt.legend()
  # plt.show()

  # # absolute min OP trough
  # op_single_peak = round(op_x[op_troughs].min(),3)
  # # first five are the lowest trough values
  # op_lowest_trough = np.array(sorted(op_x[op_troughs])[:5])
  # # average min OP trough
  # op_average_peak = round(op_lowest_trough.mean(),3)
  # print(f"OP Five Lowest Troughs:\t {op_lowest_trough} \nAverage Min OP Trough:\t {op_average_peak} cm")

  # # locate the peaks and troughs from each MA array
  # # determine the horizontal distance between peaks
  # distance = len(ma_angle) / 10
  # ma_x = np.array(ma_angle)
  # ma_peaks, _= find_peaks(ma_x, distance = distance)
  # ma_troughs, _= find_peaks(-ma_x, distance = distance)
  # plt.plot(ma_x)
  # plt.plot(ma_peaks,ma_x[ma_peaks], '^')
  # plt.plot(ma_troughs,ma_x[ma_troughs], 'v')
  # plt.title(f'MA Peak Values')
  # plt.xlabel('Frames [Hz]')
  # plt.ylabel('Angle [deg]')
  # plt.legend()
  # plt.show()

  # # absolute min MA trough
  # ma_single_peak = round(ma_x[ma_troughs].min(),3)
  # # first five are the lowest trough values
  # ma_lowest_trough = np.array(sorted(ma_x[ma_troughs])[:5])
  # # average min MA trough
  # ma_average_peak = round(ma_lowest_trough.mean(),3)
  # print(f"MA Five Lowest Trough:\t {ma_lowest_trough} \nAverage Max MA Peak:\t {ma_average_peak} cm\n")

  # # Plot OP and MA Angles together
  # plt.figure(figsize=(5,3))
  # plt.plot(op_angle, label = 'OP Cleaned')    
  # plt.plot(op_troughs, op_angle[op_troughs], "*")                
  # plt.plot(ma_angle, label = 'MA Cleaned')  
  # plt.plot(ma_troughs, ma_angle[ma_troughs], "*")                  
  # plt.legend()
  # plt.xlabel('Frames [Hz]')
  # plt.ylabel('Angle [deg]')
  # plt.title(f'OP and MA Angle {LR}.{joint}.{minmax}')
  # plt.show()


  # Method 2 --> Avg of Lowest 5% Trough Values

  # Filter
  op_angle = butter_lowpass_filter(op_angle, 1, 30, 6)
  ma_angle = butter_lowpass_filter(ma_angle, 1, 30, 6)
  # Outliers
  op_angle = remove_outliers(pd.Series(op_angle)).reset_index(drop = True)
  ma_angle = remove_outliers(pd.Series(ma_angle)).reset_index(drop = True)
  op_angle = np.array(op_angle)
  ma_angle = np.array(ma_angle)

  tmp = []
  op_troughs = []
  # absolute min OP angle
  op_single_peak = round(op_angle.min(), 3)
  # bottom 5% are the lowest angle values
  for ind,val in enumerate(op_angle): # for val in sorted(op_angle):
    if val/op_single_peak < 1.05:
      tmp.append(val)
      op_troughs.append(ind)
  op_highest_peak = np.array(tmp)
  # average min OP angle
  op_average_peak = round(op_highest_peak.mean(),3)
  # print(f'{len(op_highest_peak)} OP angles within 5% of minimum')
  # print(f"Absolute Min OP Angle:\t {op_single_peak} \nAverage Min OP Angle:\t {op_average_peak} deg")

  tmp = []
  ma_troughs = []
  # absolute min MA angle
  ma_single_peak = round(ma_angle.min(), 3)
  # bottom 5% are the lowest angle values
  for ind,val in enumerate(ma_angle): #for val in sorted(ma_angle): 
    if val/ma_single_peak < 1.05:
      tmp.append(val)
      ma_troughs.append(ind)
  ma_highest_peak = np.array(tmp)
  # average min MA angle
  ma_average_peak = round(ma_highest_peak.mean(),3)
  # print(f'{len(ma_highest_peak)} MA angles within 5% of minimum')
  # print(f"Absolute Min MA Angle:\t {ma_single_peak} \nAverage Min MA Angle:\t {ma_average_peak} deg")

  # Plot OP and MA Angles together
  plt.figure(figsize=(5,3))
  plt.plot(op_angle, label = 'OP Cleaned')    
  plt.plot(op_troughs, op_angle[op_troughs], "*")                
  plt.plot(ma_angle, label = 'MA Cleaned')  
  plt.plot(ma_troughs, ma_angle[ma_troughs], "*")                  
  plt.legend()
  plt.xlabel('Frames [Hz]')
  plt.ylabel('Angle [deg]')
  plt.title(f'OP and MA Angle {LR}.{joint}.{minmax}')
  plt.show()



  diff = op_average_peak - ma_average_peak 
  # If difference between OP and MA angles too large, take the mean of all angle values
  if diff > 10:
    op_average_peak = op_angle.mean()
    ma_average_peak = ma_angle.mean()
    diff = op_average_peak - ma_average_peak
  per = abs(op_average_peak - ma_average_peak) / abs(ma_average_peak) * 100 

  return op_average_peak, ma_average_peak, round(diff,3), round(per,3)


def max_angle(op_angle, ma_angle, LR, joint, minmax):
  ''' to compare the average angle of TOP 5% MAX Values'''

  # Method 1 --> Avg of Top 5 Peak Values

  # Filter
  op_angle = butter_lowpass_filter(op_angle, 1, 30, 6)
  ma_angle = butter_lowpass_filter(ma_angle, 1, 30, 6)
  # Outliers
  op_angle = remove_outliers(pd.Series(op_angle)).reset_index(drop = True)
  ma_angle = remove_outliers(pd.Series(ma_angle)).reset_index(drop = True)
  op_angle = np.array(op_angle)
  ma_angle = np.array(ma_angle)

  # locate the peaks and troughs from each OP array
  # determine the horizontal distance between peaks
  distance = len(op_angle) / 10
  op_x = np.array(op_angle)
  op_peaks, _= find_peaks(op_x, distance = distance)
  op_troughs, _= find_peaks(-op_x, distance = distance)
  plt.plot(op_x)
  plt.plot(op_peaks,op_x[op_peaks], '^')
  plt.plot(op_troughs,op_x[op_troughs], 'v')
  plt.title(f'OP Peak Values')
  plt.xlabel('Frames [Hz]')
  plt.ylabel('Angle [deg]')
  plt.legend()
  plt.show()

  # absolute max OP peak
  op_single_peak = round(op_x[op_peaks].max(),3)
  # last five are the highest peak values
  op_highest_peak = np.array(sorted(op_x[op_peaks])[-5:])
  # average max OP peak
  op_average_peak = round(op_highest_peak.mean(),3)
  print(f"OP Five Highest Peak:\t {op_highest_peak} \nAverage Max OP Peak:\t {op_average_peak} cm")

  # locate the peaks and troughs from each MA array
  # determine the horizontal distance between peaks
  distance = len(ma_angle) / 10
  ma_x = np.array(ma_angle)
  ma_peaks, _= find_peaks(ma_x, distance = distance)
  ma_troughs, _= find_peaks(-ma_x, distance = distance)
  plt.plot(ma_x)
  plt.plot(ma_peaks,ma_x[ma_peaks], '^')
  plt.plot(ma_troughs,ma_x[ma_troughs], 'v')
  plt.title(f'MA Peak Values')
  plt.xlabel('Frames [Hz]')
  plt.ylabel('Angle [deg]')
  plt.legend()
  plt.show()

  # absolute max MA peak
  ma_single_peak = round(ma_x[ma_peaks].max(),3)
  # last five are the highest peak values
  ma_highest_peak = np.array(sorted(ma_x[ma_peaks])[-5:])
  # average max OP peak
  ma_average_peak = round(ma_highest_peak.mean(),3)
  print(f"MA Five Highest Peak:\t {ma_highest_peak} \nAverage Max MA Peak:\t {ma_average_peak} cm\n")

  # Plot OP and MA Angles together
  plt.figure(figsize=(5,3))
  plt.plot(op_angle, label = 'OP Cleaned')    
  plt.plot(op_peaks, op_angle[op_peaks], "*")                
  plt.plot(ma_angle, label = 'MA Cleaned')  
  plt.plot(ma_peaks, ma_angle[ma_peaks], "*")                  
  plt.legend()
  plt.xlabel('Frames [Hz]')
  plt.ylabel('Angle [deg]')
  plt.title(f'OP and MA Angle {LR}.{joint}.{minmax}')
  plt.show()


  # # Method 2 --> Avg of Top 5% Peak Values

  # # Filter
  # op_angle = butter_lowpass_filter(op_angle, 1, 30, 6)
  # ma_angle = butter_lowpass_filter(ma_angle, 1, 30, 6)
  # # Outliers
  # op_angle = remove_outliers(pd.Series(op_angle)).reset_index(drop = True)
  # ma_angle = remove_outliers(pd.Series(ma_angle)).reset_index(drop = True)
  # op_angle = np.array(op_angle)
  # ma_angle = np.array(ma_angle)
  
  # tmp = []
  # op_peaks = []
  # # absolute max OP peak
  # op_single_peak = round(op_angle.max(), 3)
  # # top 5% are the highest peak values
  # for ind,val in enumerate(op_angle): # for val in sorted(op_angle):
  #   if val/op_single_peak > 0.95:
  #     tmp.append(val)
  #     op_peaks.append(ind)
  # op_highest_peak = np.array(tmp)
  # # average max OP peak
  # op_average_peak = round(op_highest_peak.mean(),3)
  # # print(f'{len(op_highest_peak)} OP angles within 5% of maximum')
  # # print(f"Absolute Max OP Peak:\t {op_single_peak} \nAverage Max OP Peak:\t {op_average_peak} deg")

  # tmp = []
  # ma_peaks = []
  # # absolute max MA peak
  # ma_single_peak = round(ma_angle.max(), 3)
  # # top 5% are the highest peak values
  # for ind,val in enumerate(ma_angle): #for val in sorted(ma_angle): 
  #   if val/ma_single_peak > 0.95:
  #     tmp.append(val)
  #     ma_peaks.append(ind)
  # ma_highest_peak = np.array(tmp)
  # # average max MA peak
  # ma_average_peak = round(ma_highest_peak.mean(),3)
  # # print(f'{len(ma_highest_peak)} MA angles within 5% of maximum')
  # # print(f"Absolute Max MA Peak:\t {ma_single_peak} \nAverage Max MA Peak:\t {ma_average_peak} deg")

  # # Plot OP and MA Angles together
  # plt.figure(figsize=(5,3))
  # plt.plot(op_angle, label = 'OP Cleaned')    
  # plt.plot(op_peaks, op_angle[op_peaks], "*")                
  # plt.plot(ma_angle, label = 'MA Cleaned')  
  # plt.plot(ma_peaks, ma_angle[ma_peaks], "*")                  
  # plt.legend()
  # plt.xlabel('Frames [Hz]')
  # plt.ylabel('Angle [deg]')
  # plt.title(f'OP and MA Angle {LR}.{joint}.{minmax}')
  # plt.show()



  diff = op_average_peak - ma_average_peak 
  # If difference between OP and MA angles too large, take the mean of all angle values
  if diff > 10:
    op_average_peak = op_angle.mean()
    ma_average_peak = ma_angle.mean()
    diff = op_average_peak - ma_average_peak
  per = abs(op_average_peak - ma_average_peak) / abs(ma_average_peak) * 100 

  return op_average_peak, ma_average_peak, round(diff,3), round(per,3)


def table_angle_results(df_list):  
  
  print(f"\nANGLE METHOD 0 - RESULTS\n")
  results = pd.concat([df_list[0],df_list[1],df_list[2],df_list[3],df_list[4],df_list[5],df_list[6],df_list[7],
                    df_list[8],df_list[9],df_list[10],df_list[11],df_list[12],df_list[13],df_list[14],df_list[15]], axis=1)

  return results


def coordinates(Ax,Ay,Az):
  '''to combine each coordinate array of a joint into one array '''

  npoints = len(Ax) 
  A = np.zeros((npoints, 3))
  for i in range(len(Ax)):
    x = Ax[i]
    y = Ay[i]
    z = Az[i]
    A[i] = x,y,z

  return A


def angle(vector_A, vector_B):
  '''to calculate the angle between two vectors using dot product'''

  # cos(alpha) = (a . b) / (|a| * |b|) --> alpha = arccos((a . b) / (|a| * |b|))

  dot_product = np.dot(vector_A, vector_B)
  prod_of_norms = np.linalg.norm(vector_A) * np.linalg.norm(vector_B)
  angle = round(np.degrees(np.arccos(dot_product / prod_of_norms)), 1)
  
  return round(dot_product, 1), angle


def angles(A, B):
  '''to get the angles at EACH FRAME of data ''' 

  angles_list = []
  for i in range(len(A)):
    dot, angles = angle(A[i], B[i])
    angles_list.append(angles)

  return np.array(angles_list)


def get_op_xyz(op_final, LeftRight, joint1, joint2, joint3):
  '''to get the segment vectors xyz coordinates relative to joint of interest'''

  # segment vector 1
  Ax = op_final[f'{joint1}{LeftRight}X'] - op_final[f'{joint2}{LeftRight}X']
  Ay = op_final[f'{joint1}{LeftRight}Y'] - op_final[f'{joint2}{LeftRight}Y']
  Az = op_final[f'{joint1}{LeftRight}Z'] - op_final[f'{joint2}{LeftRight}Z']

  # segment vector 2
  Bx = op_final[f'{joint3}{LeftRight}X'] - op_final[f'{joint2}{LeftRight}X']
  By = op_final[f'{joint3}{LeftRight}Y'] - op_final[f'{joint2}{LeftRight}Y']
  Bz = op_final[f'{joint3}{LeftRight}Z'] - op_final[f'{joint2}{LeftRight}Z']

  return Ax,Ay,Az,Bx,By,Bz


def op_joint_angle(op_final, LeftRight):
  '''to calculate the xyz for OP joints of interest --> to calculate OP joint angle'''

  # To Elbow Joint (A: Upper Arm | B: Fore Arm)
  Ax,Ay,Az,Bx,By,Bz = get_op_xyz(op_final, LeftRight, 'Shoulder', 'Elbow', 'Wrist')

  # To Shoulder Joint (C: Upper Arm | D: Trunk)
  Cx,Cy,Cz,Dx,Dy,Dz = get_op_xyz(op_final, LeftRight, 'Elbow', 'Shoulder', 'Hip')

  # To Hip Joint (E: Thigh | F: Trunk)
  Ex,Ey,Ez,Fx,Fy,Fz = get_op_xyz(op_final, LeftRight, 'Knee', 'Hip', 'Shoulder')

  # To Knee Joint (G: Thigh | H: Shank)
  Gx,Gy,Gz,Hx,Hy,Hz = get_op_xyz(op_final, LeftRight, 'Hip', 'Knee', 'Foot')

  # organize coordinates for each frame for each segment relative to joint
  A = coordinates(Ax,Ay,Az)
  B = coordinates(Bx,By,Bz)
  C = coordinates(Cx,Cy,Cz)
  D = coordinates(Dx,Dy,Dz)
  E = coordinates(Ex,Ey,Ez)
  F = coordinates(Fx,Fy,Fz)
  G = coordinates(Gx,Gy,Gz)
  H = coordinates(Hx,Hy,Hz)

  # get angles for each frame
  op_elbow_angle = angles(A,B)
  op_shoulder_angle = angles(C,D)
  op_hip_angle = angles(E,F)
  op_knee_angle = angles(G,H)

  return op_elbow_angle, op_shoulder_angle, op_hip_angle, op_knee_angle


def get_ma_xyz(ma_final, LR, joint1, joint2, joint3):
  '''to get the segment vectors xyz coordinates relative to joint of interest'''

  # segment vector 1
  Ax = ma_final[f'{LR}.{joint1}X'] - ma_final[f'{LR}.{joint2}X']
  Ay = ma_final[f'{LR}.{joint1}Y'] - ma_final[f'{LR}.{joint2}Y']
  Az = ma_final[f'{LR}.{joint1}Z'] - ma_final[f'{LR}.{joint2}Z']

  # segment vector 2
  Bx = ma_final[f'{LR}.{joint3}X'] - ma_final[f'{LR}.{joint2}X']
  By = ma_final[f'{LR}.{joint3}Y'] - ma_final[f'{LR}.{joint2}Y']
  Bz = ma_final[f'{LR}.{joint3}Z'] - ma_final[f'{LR}.{joint2}Z']

  return Ax,Ay,Az,Bx,By,Bz


def ma_joint_angle(ma_final, LR):
  '''to calculate the xyz for MA joints of interest --> to calculate MA joint angle'''

  # To Elbow Joint (A: Upper Arm | B: Fore Arm)
  Ax,Ay,Az,Bx,By,Bz = get_ma_xyz(ma_final, LR, 'Shoulder', 'Elbow', 'Wrist')

  # To Shoulder Joint (C: Upper Arm | D: Trunk)
  Cx,Cy,Cz,Dx,Dy,Dz = get_ma_xyz(ma_final, LR, 'Elbow', 'Shoulder', 'ASIS')

  # To Hip Joint (E: Thigh | F: Trunk)
  Ex,Ey,Ez,Fx,Fy,Fz = get_ma_xyz(ma_final, LR, 'Knee', 'ASIS', 'Shoulder')

  # To Knee Joint (G: Thigh | H: Shank)
  Gx,Gy,Gz,Hx,Hy,Hz = get_ma_xyz(ma_final, LR, 'ASIS', 'Knee', 'Ankle')

  # organize coordinates for each frame for each segment relative to joint
  A = coordinates(Ax,Ay,Az)
  B = coordinates(Bx,By,Bz)
  C = coordinates(Cx,Cy,Cz)
  D = coordinates(Dx,Dy,Dz)
  E = coordinates(Ex,Ey,Ez)
  F = coordinates(Fx,Fy,Fz)
  G = coordinates(Gx,Gy,Gz)
  H = coordinates(Hx,Hy,Hz)

  # get angles for each frame
  ma_elbow_angle = angles(A,B)
  ma_shoulder_angle = angles(C,D)
  ma_hip_angle = angles(E,F)
  ma_knee_angle = angles(G,H)

  return ma_elbow_angle, ma_shoulder_angle, ma_hip_angle, ma_knee_angle
