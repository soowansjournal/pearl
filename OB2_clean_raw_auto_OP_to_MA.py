# Method 3: Method 2 Fixed (Align OP to MA instead of Align MA to OP)
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

# op_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
#              'Single1', 'Single2', 'Five', 'Thirty']
# ma_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
#              'Single', 'Single', 'Five', 'Thirty']
op_games = ['BC6']
ma_games = ['BC6']

# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0601_P28']

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
                  offset_bias_x = op_joint_x - ma_joint_x      # ma_joint_x - op_joint_x
                  offset_bias_y = op_joint_y - ma_joint_z      # ma_joint_z - op_joint_y
                  offset_bias_z = op_joint_z - (-1*ma_joint_y) # -1*ma_joint_y - op_joint_z
                  offset_bias_x = remove_outliers(offset_bias_x)
                  offset_bias_y = remove_outliers(offset_bias_y)
                  offset_bias_z = remove_outliers(offset_bias_z)
                #   align_vis(offset_bias_x, op_joints[i], 'X')
                #   align_vis(offset_bias_y, op_joints[i], 'Y')
                #   align_vis(offset_bias_z, op_joints[i], 'Z')
                  offset_bias_x = offset_bias_x.mean()
                  offset_bias_y = offset_bias_y.mean()
                  offset_bias_z = offset_bias_z.mean()
                  data_vis(op_joint_x - offset_bias_x, ma_joint_x, op_joints[i] + 'X')
                  data_vis(op_joint_y - offset_bias_y, ma_joint_z, op_joints[i] + 'Y')
                  data_vis(op_joint_z - offset_bias_z, -ma_joint_y, op_joints[i] + 'Z')
                  print(f'{op_joints[i]} X: \t Offset_Bias: {round(offset_bias_x,3)}cm \tOP: {round(op_joint_x.mean(),3)}cm \tMA: {round(ma_joint_x.mean(),3)}cm')
                  print(f'{op_joints[i]} Y: \t Offset_Bias: {round(offset_bias_y,3)}cm \tOP: {round(op_joint_z.mean(),3)}cm \tMA: {round((ma_joint_y).mean(),3)}cm')
                  print(f'{op_joints[i]} Z: \t Offset_Bias: {round(offset_bias_z,3)}cm \tOP: {round(op_joint_y.mean(),3)}cm \tMA: {round(ma_joint_z.mean(),3)}cm')
                  print(f'XYZ Offset and Bias in {op_joints[i]}: {offset_bias_x} {offset_bias_y} {offset_bias_z}')

                  # Subtract from OP file 
                  for m in range(1, len(op_final.columns[:60]) - 2):
                    # Align OP coordinates to match MA (remove systemic bias)
                    if (op_joints[i]+'X') == op_final.columns[m]:
                        print(op_joints[i]+'X', op_final.columns[m])
                        # OP cm --> m
                        print("NAN?", str(offset_bias_x))
                        print("Joint:", op_joints[i])
                        print('col:', op_final.columns[m])
                        op_final.iloc[:,m] = (op_final.iloc[:,m]).astype(float) - offset_bias_x/100
                        print('col:', op_final.columns[m+1])
                        op_final.iloc[:,m+1] = (op_final.iloc[:,m+1]).astype(float) - offset_bias_y/100
                        print('col:', op_final.columns[m+2])
                        op_final.iloc[:,m+2] = (op_final.iloc[:,m+2]).astype(float) - offset_bias_z/100
                        # Convert back to string datatype
                        op_final.iloc[:,m] = (op_final.iloc[:,m]).astype(str)
                        op_final.iloc[:,m+1] = (op_final.iloc[:,m+1]).astype(str)
                        op_final.iloc[:,m+2] = (op_final.iloc[:,m+2]).astype(str)
                        print('done col:', op_final.columns[m+2])

              

                # Pre0316 (MA_X - OP_Z | MA_Y - OP_X | MA_Z - OP_Y)
                else: 
                  offset_bias_x = op_joint_x - ma_joint_y # ma_joint_x - op_joint_z
                  offset_bias_y = op_joint_y - ma_joint_z # ma_joint_y - op_joint_x
                  offset_bias_z = op_joint_z - ma_joint_x # ma_joint_z - op_joint_y
                  offset_bias_x = remove_outliers(offset_bias_x)
                  offset_bias_y = remove_outliers(offset_bias_y)
                  offset_bias_z = remove_outliers(offset_bias_z)
                  # align_vis(offset_bias_x, op_joints[i], 'X')
                  # align_vis(offset_bias_y, op_joints[i], 'Y')
                  # align_vis(offset_bias_z, op_joints[i], 'Z')
                  offset_bias_x = offset_bias_x.mean()
                  offset_bias_y = offset_bias_y.mean()
                  offset_bias_z = offset_bias_z.mean()
                  data_vis(op_joint_x - offset_bias_x, ma_joint_y, op_joints[i] + 'X')
                  data_vis(op_joint_y - offset_bias_y, ma_joint_z, op_joints[i] + 'Y')
                  data_vis(op_joint_z - offset_bias_z, ma_joint_x, op_joints[i] + 'Z')
                  print(f'{op_joints[i]} X: \t Offset_Bias: {round(offset_bias_x,3)}cm \tOP: {round(op_joint_x.mean(),3)}cm \tMA: {round(ma_joint_x.mean(),3)}cm')
                  print(f'{op_joints[i]} Y: \t Offset_Bias: {round(offset_bias_y,3)}cm \tOP: {round(op_joint_z.mean(),3)}cm \tMA: {round((ma_joint_y).mean(),3)}cm')
                  print(f'{op_joints[i]} Z: \t Offset_Bias: {round(offset_bias_z,3)}cm \tOP: {round(op_joint_y.mean(),3)}cm \tMA: {round(ma_joint_z.mean(),3)}cm')
                  print(f'XYZ Offset and Bias in {op_joints[i]}: {offset_bias_x} {offset_bias_y} {offset_bias_z}')

                  # Subtract from OP file 
                  for m in range(1, len(op_final.columns[:60]) - 2):
                    # Align OP coordinates to match MA (remove systemic bias)
                    if (op_joints[i]+'X') == op_final.columns[m]:
                        print(op_joints[i]+'X', op_final.columns[m])
                        # OP cm --> m
                        print("NAN?", str(offset_bias_x))
                        print("Joint:", op_joints[i])
                        print('col:', op_final.columns[m])
                        op_final.iloc[:,m] = (op_final.iloc[:,m]).astype(float) - offset_bias_x/100
                        print('col:', op_final.columns[m+1])
                        op_final.iloc[:,m+1] = (op_final.iloc[:,m+1]).astype(float) - offset_bias_y/100
                        print('col:', op_final.columns[m+2])
                        op_final.iloc[:,m+2] = (op_final.iloc[:,m+2]).astype(float) - offset_bias_z/100
                        # Convert back to string datatype
                        op_final.iloc[:,m] = (op_final.iloc[:,m]).astype(str)
                        op_final.iloc[:,m+1] = (op_final.iloc[:,m+1]).astype(str)
                        op_final.iloc[:,m+2] = (op_final.iloc[:,m+2]).astype(str)
                        print('done col:', op_final.columns[m+2])
                  

                


    
            
           









              # # Visualize Data
              # x = np.linspace(0,len(op_final),len(op_final))  
              # plt.figure(figsize=(5,3))                                    
              # plt.plot(x, op_final['WristLeftX'], label = 'OP Synch')                    
              # plt.title(f'OP Left Wrist Horizontal')
              # plt.xlabel('Frames [Hz]')
              # plt.ylabel('Distance [cm]')
              # plt.show()

              # x = np.linspace(0,len(ma_final),len(ma_final))  
              # plt.figure()                                 
              # plt.plot(x, ma_final['L.Wrist'], label = 'MA Synch')                     
              # plt.title(f'MA Left Wrist Horizontal')
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
              if 'BC' in op_games[game_ind]:
                # DOWNLOAD CLEANED OP DATA
                op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                # DOWNLOAD CLEANED MA BOOT CAMP DATA
                ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)
              else:
                op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                # DOWNLOAD CLEANED MA BOOT CAMP DATA
                ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)
                 

            #   # DOWNLOAD FILES TO SPECIFIC LOCATION
            #   if 'BC' in op_games[game_ind] and op_game[4:] in count:
            #     op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Count/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
            #     ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Count/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
            #   elif 'BC' in op_games[game_ind] and op_game[4:] in timer:
            #     op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Timer/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
            #     ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Timer/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
            #   elif op_games[game_ind] == 'Single1':
            #     op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleR/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
            #     ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleR/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
            #   elif op_games[game_ind] == 'Single2':
            #     op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleL/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
            #     ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleL/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
            #   elif op_games[game_ind] == 'Five':
            #     op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Five/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
            #     ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Five/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)        
            #   elif op_games[game_ind] == 'Thirty':
            #     op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Thirty/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
            #     ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Thirty/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)        

print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)

print("\nFOLLOWING GAMES HAVE UNKNOWN PEAKS:", game_peaks_unknown)
