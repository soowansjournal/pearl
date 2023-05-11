'''
---
# **OB1 AutoAnalysis 1-5) Maximum/Mean Hand Speed**
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


def plot_speed(op_array, ma_array, allvspos, rightorleft):

  plt.figure(figsize=(5,3))
  plt.plot(op_array, label = 'OP Velocity')
  plt.plot(ma_array, label = 'MA Velocity')
  plt.title(f'{allvspos} {rightorleft} Hand Speed')
  plt.xlabel('Frames')
  plt.ylabel('Speed [cm/s]')
  plt.legend()
  plt.show()

  return


def table_vel(op_vel, ma_vel, diff_vel, per_vel, rightorleft, maxormean, order = 'NA'):
  
  if order != 'NA':
    vel_table = pd.DataFrame()
    vel_table['Order'] = order
    vel_table[f'OP({rightorleft}.{maxormean})'] = op_vel
    vel_table[f'MA({rightorleft}.{maxormean})'] = ma_vel
    vel_table['Diff[cm/s]'] = diff_vel
    vel_table['Error[%]'] = per_vel
  else:
    single_vals = [op_vel, ma_vel, diff_vel, per_vel]
    col = [f'OP({rightorleft}.{maxormean})', f'MA({rightorleft}.{maxormean})', 'Diff[cm/s]', 'Error[%]']
    vel_table = pd.DataFrame([single_vals], columns = col)

  return vel_table


def table_vel_results(method, left_vel_max, left_vel_mean, right_vel_max, right_vel_mean):  
  
  print(f"\nREACH METHOD {method} - RESULTS\n")
  results = pd.concat([left_vel_max, left_vel_mean, right_vel_max, right_vel_mean], axis=1)

  return results


def get_speed(array_test, op_ma_times):

  all_vel = []

  # instantaneous velocity
  for i in range(len(array_test) - 1):
    position = array_test[i+1] - array_test[i]
    delta_time = op_ma_times[i+1] - op_ma_times[i]
    velocity = position / delta_time
    all_vel.append(velocity)

  # filter the velocity using a butterworth low pass filter
  y = np.array(all_vel)
  frames = len(op_ma_times)
  seconds = (op_ma_times.iloc[-1] - op_ma_times.iloc[0]) 
  op_hz = frames /seconds
  all_vel = butter_lowpass_filter(y,0.5,op_hz,4) # (data, cutoff, fs, order)

  return all_vel


def speed_calculations(op_final, ma_final):

  op_ma_times = op_final.Time

  # OP Velocity 

  # OP LEFT
  op_array_test = np.array(np.sqrt(op_final['WristLeftY']**2 + op_final['WristLeftZ']**2 + op_final['WristLeftX']**2))
  op_all_left = get_speed(op_array_test, op_ma_times)

  # OP RIGHT
  op_array_test = np.array(np.sqrt(op_final['WristRightY']**2 + op_final['WristRightZ']**2 + op_final['WristRightX']**2))
  op_all_right = get_speed(op_array_test, op_ma_times)


  # MA Velocity 

  # MA LEFT
  ma_array_test = np.array(np.sqrt(ma_final['L.WristY']**2 + ma_final['L.WristZ']**2 + ma_final['L.WristX']**2))
  ma_all_left = get_speed(ma_array_test, op_ma_times)

  # MA RIGHT
  ma_array_test = np.array(np.sqrt(ma_final['R.WristY']**2 + ma_final['R.WristZ']**2 + ma_final['R.WristX']**2))
  ma_all_right = get_speed(ma_array_test, op_ma_times)


  # plot velocity 
  plot_speed(op_all_left, ma_all_left, 'Instantaneous Velocity', 'LEFT')
  plot_speed(op_all_right, ma_all_right, 'Instantaneous Velocity', 'RIGHT')


  return op_all_left, ma_all_left, op_all_right, ma_all_right


def peaks_finder(op_ma, order, side, array_test, reachvsspeed):

  # determine the horizontal distance between peaks
  distance = len(array_test) / 10

  x = np.array(array_test)
  peaks, _= find_peaks(x, distance = distance)
  troughs, _= find_peaks(-x, distance = distance)
  plt.plot(x, label = op_ma)
  plt.plot(peaks,x[peaks], '^')
  plt.plot(troughs,x[troughs], 'v')
  plt.title(f'{op_ma} Peak Values from {side} {reachvsspeed} {order}')
  plt.xlabel('Frames [Hz]')
  plt.ylabel('Distance [cm]')
  plt.legend()
  plt.show()

  print(f"{op_ma} peaks: {len(peaks)} {op_ma} troughs: {len(troughs)}")

  return x, peaks, troughs


def speed_max(op_array, ma_array, leftorright):


  op_array = np.array(op_array)
  ma_array = np.array(ma_array)


  # speed_max --> --> Avg of Top 5 Peak Values

  # find the max velocity by averaging the peaks troughs 
  op_x, op_peaks, op_troughs = peaks_finder('OP', '3D', leftorright, op_array, 'Velocity')
  op_max = (abs(op_x[op_peaks]).mean() + abs(op_x[op_troughs]).mean() ) / 2

  ma_x, ma_peaks, ma_troughs = peaks_finder('MA', '3D', leftorright, ma_array, 'Velocity')
  ma_max = (abs(ma_x[ma_peaks]).mean() + abs(ma_x[ma_troughs]).mean() ) / 2
  
  print(f"OP Max (avg of top 5 values): {op_max}")
  print(f"MA Max (avg of top 5 values): {ma_max}")


  # speed_max() --> Avg of Within 5% Peak Values

  tmp = []
  # absolute max OP peak
  op_single_peak = op_array.max()
  print(f"OP Max (absolute): {op_single_peak}")
  # within 5% are the highest peak values
  for val in op_array:
    #if val / op_single_peak > 0.95:
    if abs(op_single_peak - val) / op_single_peak < 0.10:
      tmp.append(val)
  op_highest_peak = np.array(tmp)
  # average max OP peak
  op_max = round(op_highest_peak.mean(),3)
  print(f"OP Max (avg of top 5% values): {op_max}")

  tmp = []
  # absolute max MA peak
  ma_single_peak = ma_array.max()
  print(f"MA Max (absolute): {ma_single_peak}")
  # within 5% are the highest peak values
  for val in sorted(ma_array):
    #if val / ma_single_peak > 0.95:
    if abs(ma_single_peak - val) / ma_single_peak < 0.10:
      tmp.append(val)
  ma_highest_peak = np.array(tmp)
  # average max MA peak
  ma_max = round(ma_highest_peak.mean(),3)
  print(f"MA Max (avg of top 5% values): {ma_max}")


  diff_max = op_max - ma_max
  per_max = abs(op_max - ma_max) / ma_max * 100


  return round(op_max,3), round(ma_max,3), round(diff_max,3), round(per_max,3), op_highest_peak, ma_highest_peak


def speed_mean(op_array, ma_array):

  op_array = np.array(op_array)
  ma_array = np.array(ma_array)

  # continue calculation
  op_mean = abs(op_array).mean()
  ma_mean = abs(ma_array).mean()
  diff_mean = op_mean - ma_mean
  per_mean = abs(op_mean - ma_mean) / ma_mean * 100

  return round(op_mean,3), round(ma_mean,3), round(diff_mean,3), round(per_mean,3)