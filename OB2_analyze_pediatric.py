'''
---
# **OB1 AutoAnalysis 1-4) Extent of Hand Reach**
---

### **_Use for all ANALYSIS since cleaned data is all SAME FORMAT_**

# Steps
# 1) For each game
# 2) For each participant
# 3) For each side
# 4) For each coordinate
# 5) OP vs MA
# 6) Hand Reach
# 7) Max Hand Reach

Soowan Choi
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# to find peaks in data for synchronization
from scipy.signal import find_peaks


def peaks_method2(order, side, op_array_test, ma_array_test):
        
        def peaks_finder(op_ma, order, side, array_test, reachvsspeed):

            # determine the horizontal distance between peaks
            distance = len(op_final) / 10

            x = np.array(array_test)
            peaks, _= find_peaks(x, distance = distance)
            troughs, _= find_peaks(-x, distance = distance)
            # plt.plot(x, label = op_ma)
            # plt.plot(peaks,x[peaks], '^')
            # plt.plot(troughs,x[troughs], 'v')
            # plt.title(f'{op_ma} Peak Values from {side} {reachvsspeed} {order}')
            # plt.xlabel('Frames [Hz]')
            # plt.ylabel('Distance [cm]')
            # plt.legend()
            # plt.show()

            print(f"{op_ma} peaks: {len(peaks)} {op_ma} troughs: {len(troughs)}")

            return x, peaks, troughs

        # locate peaks and troughs from OP reach array
        op_x, op_peaks, op_troughs = peaks_finder('OP', order, side, op_array_test, 'Reach')

        # locate peaks and troughs from MA reach array
        ma_x, ma_peaks, ma_troughs = peaks_finder('MA', order, side, ma_array_test, 'Reach')

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


        # peaks_method2() --> Avg of Top 5 Peak Values

        # absolute max OP peak
        op_single_peak = round(op_x[op_peaks].max(),3)
        # last five are the highest peak values
        op_highest_peak = np.array(sorted(op_x[op_peaks])) # [-5:]
        # average max OP peak
        op_average_peak = round(op_highest_peak.mean(),3)
        print(f"OP Five Highest Peak:\t {op_highest_peak} \nAverage Max OP Peak:\t {op_average_peak} cm")

        # absolute max MA peak
        ma_single_peak = round(ma_x[op_peaks].max(),3)
        # last five are the highest peak values
        ma_highest_peak = np.array(sorted(ma_x[op_peaks])) # [-5:]
        # average max OP peak
        ma_average_peak = round(ma_highest_peak.mean(),3)
        print(f"MA Five Highest Peak:\t {ma_highest_peak} \nAverage Max MA Peak:\t {ma_average_peak} cm\n")





        # # peaks_method2() --> Avg of Top 5% Peak Values

        # tmp = []
        # op_peaks = []
        # # absolute max OP peak
        # op_single_peak = op_array_test.max()
        # # top 5% are the highest peak values
        # for ind,val in enumerate(op_array_test):
        #     if val/op_single_peak > 0.95:
        #         tmp.append(val)
        #         op_peaks.append(ind)
        # op_highest_peak = np.array(tmp)
        # # average max OP peak
        # op_average_peak = round(op_highest_peak.mean(),3)
        # print(f"Absolute Max OP Peak:\t {op_single_peak} \nAverage Max OP Peak:\t {op_average_peak} cm")

        # tmp = []
        # ma_peaks = []
        # # absolute max MA peak
        # ma_single_peak = ma_array_test.max()
        # # top 5% are the highest peak values
        # for ind,val in enumerate(ma_array_test):
        #     if val/ma_single_peak > 0.95:
        #         tmp.append(val)
        #         ma_peaks.append(ind)
        # ma_highest_peak = np.array(tmp)
        # # average max MA peak
        # ma_average_peak = round(ma_highest_peak.mean(),3)
        # print(f"Absolute Max MA Peak:\t {ma_single_peak} \nAverage Max MA Peak:\t {ma_average_peak} cm")


        # # plot the peak values found from MA reach data
        # plt.figure(figsize=(5,3))
        # plt.plot(op_array_test, label = 'OP Cleaned')    
        # plt.plot(op_array_test[op_peaks], "*")                
        # plt.plot(ma_array_test, label = 'MA Cleaned')  
        # plt.plot(ma_array_test[ma_peaks], "*")                  
        # plt.legend()
        # plt.xlabel('Frames [Hz]')
        # plt.ylabel('Distance [cm]')
        # plt.title('OP and MA Reach Peaks - SEPARATE')
        # plt.xlabel('Frames [Hz]')
        # plt.ylabel('Distance [cm]')
        # plt.show()

        return op_average_peak, ma_average_peak, op_highest_peak, ma_highest_peak



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



def reach_calculations(op_final, ma_final):
  # OP Extent of Hand Reach

  # OP LEFT
  # left hand joint = (hx,hy,hz)
  op_le_hx = op_final['WristLeftX'] - np.mean(op_final['WristLeftX'])
  op_le_hy = op_final['WristLeftY'] - np.mean(op_final['WristLeftY'])
  op_le_hz = op_final['WristLeftZ'] - np.mean(op_final['WristLeftZ'])
  # 3D
  leftreach_op = np.sqrt((op_le_hx)**2 + (op_le_hy)**2 + (op_le_hz)**2)
  # x,y,z
  leftreachx_op = abs(op_le_hx)
  leftreachy_op = abs(op_le_hy)
  leftreachz_op = abs(op_le_hz)

  # OP RIGHT 
  # right hand joint = (hx,hy,hz)
  op_ri_hx = op_final['WristRightX'] - np.mean(op_final['WristRightX'])
  op_ri_hy = op_final['WristRightY'] - np.mean(op_final['WristRightY'])
  op_ri_hz = op_final['WristRightZ'] - np.mean(op_final['WristRightZ'])
  # 3D
  rightreach_op = np.sqrt((op_ri_hx)**2 + (op_ri_hy)**2 + (op_ri_hz)**2)
  # x,y,z
  rightreachx_op = abs(op_ri_hx)
  rightreachy_op = abs(op_ri_hy)
  rightreachz_op = abs(op_ri_hz)


  # MA LEFT 
  # left hand joint = (hx,hy,hz)
  ma_le_hx = ma_final['L.WristX'] - np.mean(ma_final['L.WristX'])
  ma_le_hy = ma_final['L.WristY'] - np.mean(ma_final['L.WristY'])
  ma_le_hz = ma_final['L.WristZ'] - np.mean(ma_final['L.WristZ'])
  # 3D
  leftreach_ma = np.sqrt((ma_le_hx)**2 + (ma_le_hy)**2 + (ma_le_hz)**2)
  # x,y,z
  leftreachx_ma = abs(ma_le_hx)
  leftreachy_ma = abs(ma_le_hy)
  leftreachz_ma = abs(ma_le_hz)

  # MA RIGHT
  # right hand joint = (hx,hy,hz)
  ma_ri_hx = ma_final['R.WristX'] - np.mean(ma_final['R.WristX'])
  ma_ri_hy = ma_final['R.WristY'] - np.mean(ma_final['R.WristY'])
  ma_ri_hz = ma_final['R.WristZ'] - np.mean(ma_final['R.WristZ'])
  # 3D
  rightreach_ma = np.sqrt((ma_ri_hx)**2 + (ma_ri_hy)**2 + (ma_ri_hz)**2)
  # x,y,z
  rightreachx_ma = abs(ma_ri_hx)
  rightreachy_ma = abs(ma_ri_hy)
  rightreachz_ma = abs(ma_ri_hz)

  # LEFT REACH DATA
  op_array_left = [leftreachx_op, leftreachy_op, leftreachz_op, leftreach_op]
  ma_array_left = [leftreachx_ma, leftreachy_ma, leftreachz_ma, leftreach_ma]

  # RIGHT REACH DATA
  op_array_right = [rightreachx_op, rightreachy_op, rightreachz_op, rightreach_op]
  ma_array_right = [rightreachx_ma, rightreachy_ma, rightreachz_ma, rightreach_ma]

  return op_array_left, ma_array_left, op_array_right, ma_array_right



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



def table_reach_results(method, left_reach_max, right_reach_max, left_reach_mean = pd.Series(['NA','NA','NA','NA']), right_reach_mean = pd.Series(['NA','NA','NA','NA'])):  
  
  print(f"\nREACH METHOD {method} - RESULTS\n")
  results = pd.concat([left_reach_max, right_reach_max, left_reach_mean, right_reach_mean], axis=1)

  return results





def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


# Select Game
# op_games = ['Pediatric']
# ma_games = ['Pediatric']
op_games = ['Pediatric']
ma_games = ['Pediatric']


# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']


# 1) For each game
for game_ind in range(len(op_games)):

    directory_unknown =[]
    data = []

    # 2) For each participant
    for mmdd_p in mmdd_p_all:

        print(f'\n\n\n\n\n\n\n\n{mmdd_p}\n\n\n\n\n\n\n\n')
        op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + op_games[game_ind] + "-Data-OP-CLEAN.csv"
        ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + ma_games[game_ind] + "-MA-CLEAN.csv"
        #print(op_file, '\t', ma_file)

        try: 
            # If Cleaned Data: OB1_clean_redo.py --> Load Files from "Auto_Clean_" instead of "Clean_"
            # Load OP Data
            op = load_op('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/Auto_Clean_' + mmdd_p + '/' + op_file)
            print(op.head(3))

            # Load MA Data
            ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/Auto_Clean_' + mmdd_p + '/' + ma_file)
            print(ma.head(3))

        except FileNotFoundError:
                # if directory game file doesn't exist, go to next game
                directory_unknown.append(op_file)
                continue



        op_align_joints = op.copy()
        ma_align_joints = ma.copy()


        # # Visualize ALL Data (39 graphs total)
        # op_head = ['Head']
        # ma_head = ['Front.Head']
        # op_side = ['Left','Right']
        # ma_side = ['L.','R.']
        # xyz = ['Y','Z','X']
        # # Head Data
        # for i in range(len(op_head)):
        #     for k in range(len(xyz)):
        #         op_joint = op_head[i] + xyz[k]
        #         ma_joint = ma_head[i] + xyz[k]
        #         joint = ma_joint
        #         if xyz[k] == 'Y':
        #             data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align horizontal(Y) coordinate
        #         elif xyz[k] == 'Z':
        #             data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align vertical(Z) coordinate
        #         elif xyz[k] == 'X':
        #             data_vis(op_align_joints, ma_align_joints, joint, op_joint, ma_joint)  # align depth(X) coordinate


        op_final = op_align_joints.copy().reset_index().drop("index",axis=1)
        ma_final = ma_align_joints.copy().reset_index().drop("index",axis=1)



        # 1-4) Extent of Hand Reach

        # 3) For each side
        # 4) For each coordinate
        # 5) OP vs MA
        # 6) Hand Reach
        # 7) Max Hand Reach

        op_array_left, ma_array_left, op_array_right, ma_array_right = reach_calculations(op_final, ma_final)

        # left reach lists (remove outliers)
        side = 'Left'
        op_left_max, ma_left_max, diff_left_max, per_left_max, op_highest_leftreach, ma_highest_leftreach = reach_lists(side, op_array_left, ma_array_left)

        # right reach lists (remove outliers)
        side = 'Right'
        op_right_max, ma_right_max, diff_right_max, per_right_max, op_highest_rightreach, ma_highest_rightreach = reach_lists(side, op_array_right, ma_array_right)

        # show results
        order = ['X', 'Y', 'Z', '3D']

        left_reach_max = table_left_max(order, op_left_max, ma_left_max, diff_left_max, per_left_max)
        right_reach_max = table_right_max(order, op_right_max, ma_right_max, diff_right_max, per_right_max)

        reach = table_reach_results(2, left_reach_max, right_reach_max)

        # DOWNLOAD the reach results --> paste into data results
        #reach.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-reach.csv', encoding = 'utf-8-sig') 


        # Reorganize DataFrame Results
        dataframes = []
        side_coord = ['L.Max.X', 'L.Max.Y', 'L.Max.Z', 'L.Max.3D', 'R.Max.X', 'R.Max.Y', 'R.Max.Z', 'R.Max.3D']
        specify = ['(OP)', '(MA)', '(Diff)', '(Error%)']
        new_col = []

        # Create New Column Names
        for i in side_coord:
            for j in specify:
                name = i + j
                new_col.append(name)

        # Recreate into single row
        for i in range(len(left_reach_max)):
            left = pd.DataFrame(np.array(left_reach_max.iloc[i,1:]))
            dataframes.append(left.transpose())
            
        for i in range(len(right_reach_max)):
            right = pd.DataFrame(np.array(right_reach_max.iloc[i,1:]))
            dataframes.append(right.transpose())


        try:
            reach_re = pd.concat(dataframes, axis = 1)
        except:
            continue
        reach_re = np.array(reach_re[0:])
        reach_re = pd.DataFrame(reach_re, columns = new_col, index = [mmdd_p[-3:]])


        # DOWNLOAD the reach results --> paste into data results
        reach_re.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Pediatric/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-reach.csv', encoding = 'utf-8-sig') 

        data.append(reach_re)

        print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)
    

    # if game doesn't exist for this participant
    try:
        reach_overall = pd.concat(data)
    except:
        continue

    # DOWNLOAD the OVERALL Max Hand Reach Values --> paste into data results
    reach_overall.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Pediatric/2023-{op_games[game_ind]}-reach.csv', encoding = 'utf-8-sig') 









       