'''
---
# **OB1 AutoAnalysis 1-4) Extent of Hand Reach**
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


def table_left_max(order, op_left_max, ma_left_max, diff_left_max, per_left_max):
  # LEFT REACH MAX (X,Y,Z,3D)
  left_reach_max = pd.DataFrame()

  left_reach_max['Order'] = order
  left_reach_max['OP(L.MAX)'] = op_left_max
  left_reach_max['MA(L.MAX)'] = ma_left_max
  left_reach_max['Diff[cm]'] = diff_left_max
  left_reach_max['Error[%]'] = per_left_max

  return left_reach_max


def table_right_max(order, op_right_max, ma_right_max, diff_right_max, per_right_max):
  # RIGHT REACH MAX (X,Y,Z,3D)
  right_reach_max = pd.DataFrame()

  right_reach_max['Order'] = order
  right_reach_max['OP(R.MAX)'] = op_right_max
  right_reach_max['MA(R.MAX)'] = ma_right_max
  right_reach_max['Diff[cm]'] = diff_right_max
  right_reach_max['Error[%]'] = per_right_max

  return right_reach_max


def table_left_mean(order, op_left_mean, ma_left_mean, diff_left_mean, per_left_mean):
  # LEFT REACH MEAN (X,Y,Z,3D)
  left_reach_mean = pd.DataFrame()

  left_reach_mean['Order'] = order
  left_reach_mean['OP(L.MEAN)'] = op_left_mean
  left_reach_mean['MA(L.MEAN)'] = ma_left_mean
  left_reach_mean['Diff[cm]'] = diff_left_mean
  left_reach_mean['Error[%]'] = per_left_mean

  return left_reach_mean


def table_right_mean(order, op_right_mean, ma_right_mean, diff_right_mean, per_right_mean):
  # RIGHT REACH MEAN (X,Y,Z,3D)
  right_reach_mean = pd.DataFrame()

  right_reach_mean['Order'] = order
  right_reach_mean['OP(R.MEAN)'] = op_right_mean
  right_reach_mean['MA(R.MEAN)'] = ma_right_mean
  right_reach_mean['Diff[cm]'] = diff_right_mean
  right_reach_mean['Error[%]'] = per_right_mean 

  return right_reach_mean


def table_reach_results(method, left_reach_max, right_reach_max, left_reach_mean = pd.Series(['NA','NA','NA','NA']), right_reach_mean = pd.Series(['NA','NA','NA','NA'])):  
  
  print(f"\nREACH METHOD {method} - RESULTS\n")
  results = pd.concat([left_reach_max, right_reach_max, left_reach_mean, right_reach_mean], axis=1)

  return results


def reach_calculations(op_final, ma_final):
  # OP Extent of Hand Reach

  # shoulder centre = (sx,sy,sz) 
  # op_final['SpineShoulderX'] 
  op_sx = ((op_final['ShoulderRightX'] + op_final['ShoulderLeftX']) / 2)  
  op_sy = ((op_final['ShoulderRightY'] + op_final['ShoulderLeftY']) / 2)
  op_sz = ((op_final['ShoulderRightZ'] + op_final['ShoulderLeftZ']) / 2)

  # OP LEFT
  # left hand joint = (hx,hy,hz)
  op_le_hx = op_final['WristLeftX']
  op_le_hy = op_final['WristLeftY']
  op_le_hz = op_final['WristLeftZ']
  # 3D
  leftreach_op = np.sqrt((op_le_hx - op_sx)**2 + (op_le_hy - op_sy)**2 + (op_le_hz - op_sz)**2)
  # x,y,z
  leftreachx_op = abs(op_le_hx - op_sx)
  leftreachy_op = abs(op_le_hy - op_sy)
  leftreachz_op = abs(op_le_hz - op_sz)

  # OP RIGHT 
  # right hand joint = (hx,hy,hz)
  op_ri_hx = op_final['WristRightX']
  op_ri_hy = op_final['WristRightY']
  op_ri_hz = op_final['WristRightZ']
  # 3D
  rightreach_op = np.sqrt((op_ri_hx - op_sx)**2 + (op_ri_hy - op_sy)**2 + (op_ri_hz - op_sz)**2)
  # x,y,z
  rightreachx_op = abs(op_ri_hx - op_sx)
  rightreachy_op = abs(op_ri_hy - op_sy)
  rightreachz_op = abs(op_ri_hz - op_sz)


  # MA Extent of Hand Reach

  # shoulder centre = (sx,sy,sz)
  ma_sx = ((ma_final['R.ShoulderX'] + ma_final['L.ShoulderX']) / 2)
  ma_sy = ((ma_final['R.ShoulderY'] + ma_final['L.ShoulderY']) / 2)
  ma_sz = ((ma_final['R.ShoulderZ'] + ma_final['L.ShoulderZ']) / 2)

  # MA LEFT 
  # left hand joint = (hx,hy,hz)
  ma_le_hx = ma_final['L.WristX']
  ma_le_hy = ma_final['L.WristY']
  ma_le_hz = ma_final['L.WristZ']
  # 3D
  leftreach_ma = np.sqrt((ma_le_hx - ma_sx)**2 + (ma_le_hy - ma_sy)**2 + (ma_le_hz - ma_sz)**2)
  # x,y,z
  leftreachx_ma = abs(ma_le_hx - ma_sx)
  leftreachy_ma = abs(ma_le_hy - ma_sy)
  leftreachz_ma = abs(ma_le_hz - ma_sz)

  # MA RIGHT
  # right hand joint = (hx,hy,hz)
  ma_ri_hx = ma_final['R.WristX']
  ma_ri_hy = ma_final['R.WristY']
  ma_ri_hz = ma_final['R.WristZ']
  # 3D
  rightreach_ma = np.sqrt((ma_ri_hx - ma_sx)**2 + (ma_ri_hy - ma_sy)**2 + (ma_ri_hz - ma_sz)**2)
  # x,y,z
  rightreachx_ma = abs(ma_ri_hx - ma_sx)
  rightreachy_ma = abs(ma_ri_hy - ma_sy)
  rightreachz_ma = abs(ma_ri_hz - ma_sz)

  # LEFT REACH DATA
  op_array_left = [leftreachx_op, leftreachy_op, leftreachz_op, leftreach_op]
  ma_array_left = [leftreachx_ma, leftreachy_ma, leftreachz_ma, leftreach_ma]

  # RIGHT REACH DATA
  op_array_right = [rightreachx_op, rightreachy_op, rightreachz_op, rightreach_op]
  ma_array_right = [rightreachx_ma, rightreachy_ma, rightreachz_ma, rightreach_ma]

  return op_array_left, ma_array_left, op_array_right, ma_array_right


def peaks_finder(op_ma, order, side, array_test, reachvsspeed):

  # determine the horizontal distance between peaks
  distance = len(op_final) / 10

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


def peaks_method2(order, side, op_array_test, ma_array_test):

        # # locate peaks and troughs from OP reach array
        # op_x, op_peaks, op_troughs = peaks_finder('OP', order, side, op_array_test, 'Reach')

        # # locate peaks and troughs from MA reach array
        # ma_x, ma_peaks, ma_troughs = peaks_finder('MA', order, side, ma_array_test, 'Reach')

        # # plot the peak values found from MA reach data
        # plt.plot(op_array_test, label = 'OP Cleaned')    
        # plt.plot(op_array_test[op_peaks], "*")                
        # plt.plot(ma_array_test, label = 'MA Cleaned')  
        # plt.plot(ma_array_test[op_peaks], "*")                  
        # plt.legend()
        # plt.xlabel('Frames [Hz]')
        # plt.ylabel('Distance [cm]')
        # plt.title('OP and MA Reach Peaks - SEPARATE')
        # plt.xlabel('Frames [Hz]')
        # plt.ylabel('Distance [cm]')
        # plt.show()


        # # # remove outliers
        # # op_array_test = remove_outliers(pd.Series(op_array_test))
        # # ma_array_test = remove_outliers(pd.Series(ma_array_test))


        # # peaks_method2() --> Avg of Top 5 Peak Values

        # # absolute max OP peak
        # op_single_peak = round(op_x[op_peaks].max(),3)
        # # last five are the highest peak values
        # op_highest_peak = np.array(sorted(op_x[op_peaks])[-5:])
        # # average max OP peak
        # op_average_peak = round(op_highest_peak.mean(),3)
        # print(f"OP Five Highest Peak:\t {op_highest_peak} \nAverage Max OP Peak:\t {op_average_peak} cm")

        # # absolute max MA peak
        # ma_single_peak = round(ma_x[op_peaks].max(),3)
        # # last five are the highest peak values
        # ma_highest_peak = np.array(sorted(ma_x[op_peaks])[-5:])
        # # average max OP peak
        # ma_average_peak = round(ma_highest_peak.mean(),3)
        # print(f"MA Five Highest Peak:\t {ma_highest_peak} \nAverage Max MA Peak:\t {ma_average_peak} cm\n")





        # peaks_method2() --> Avg of Top 5% Peak Values

        tmp = []
        op_peaks = []
        # absolute max OP peak
        op_single_peak = op_array_test.max()
        # top 5% are the highest peak values
        for ind,val in enumerate(op_array_test):
            if val/op_single_peak > 0.95:
                tmp.append(val)
                op_peaks.append(ind)
        op_highest_peak = np.array(tmp)
        # average max OP peak
        op_average_peak = round(op_highest_peak.mean(),3)
        print(f"Absolute Max OP Peak:\t {op_single_peak} \nAverage Max OP Peak:\t {op_average_peak} cm")

        tmp = []
        ma_peaks = []
        # absolute max MA peak
        ma_single_peak = ma_array_test.max()
        # top 5% are the highest peak values
        for ind,val in enumerate(ma_array_test):
            if val/ma_single_peak > 0.95:
                tmp.append(val)
                ma_peaks.append(ind)
        ma_highest_peak = np.array(tmp)
        # average max MA peak
        ma_average_peak = round(ma_highest_peak.mean(),3)
        print(f"Absolute Max MA Peak:\t {ma_single_peak} \nAverage Max MA Peak:\t {ma_average_peak} cm")


        # plot the peak values found from MA reach data
        plt.figure(figsize=(5,3))
        plt.plot(op_array_test, label = 'OP Cleaned')    
        plt.plot(op_array_test[op_peaks], "*")                
        plt.plot(ma_array_test, label = 'MA Cleaned')  
        plt.plot(ma_array_test[ma_peaks], "*")                  
        plt.legend()
        plt.xlabel('Frames [Hz]')
        plt.ylabel('Distance [cm]')
        plt.title('OP and MA Reach Peaks - SEPARATE')
        plt.xlabel('Frames [Hz]')
        plt.ylabel('Distance [cm]')
        plt.show()

        return op_average_peak, ma_average_peak, op_highest_peak, ma_highest_peak


def reach_lists(side, op_array_left, ma_array_left):

  # LEFT SIDE AS AN EXAMPLE

  # left reach lists 
  op_left_max = []
  ma_left_max = []
  diff_left_max = []
  per_left_max = []

  side = side

  for i in range(len(ma_array_left)):
    order = ['X', 'Y', 'Z', '3D']

    # left max
    op_peak_val, ma_peak_val, op_highest_peak, ma_highest_peak = peaks_method2(order[i], side, op_array_left[i], ma_array_left[i])
    
    # continue calculation
    op_left_max.append(op_peak_val)
    ma_left_max.append(ma_peak_val)
    diff_left_max.append(op_peak_val - ma_peak_val)
    per = round(abs(op_peak_val - ma_peak_val) / abs(ma_peak_val) * 100,3)
    per_left_max.append(per)

  return op_left_max, ma_left_max, diff_left_max, per_left_max, op_highest_peak, ma_highest_peak