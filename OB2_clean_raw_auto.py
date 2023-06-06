'''
---
# **Copy of CHAPTER 3: OB2 Clean Raw Auto**
---

Soowan Choi
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
op_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
             'Single1', 'Single2', 'Five', 'Thirty']
ma_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
             'Single', 'Single', 'Five', 'Thirty']

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
              print(peaks[op_games[game_ind]][i:i+7])
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
                                        op_game = op_games[game_ind] + '-' + 'NA'
                                        ma_game = ma_games[game_ind] + '-' + 'NA'
                                        break


              # # DOWNLOAD FILES TO DOWNLOADS FOLDER
              # if 'BC' in op_games[game_ind]:
              #   # DOWNLOAD CLEANED OP DATA
              #   op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)
              # else:
              #   op_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
              #   # DOWNLOAD CLEANED MA BOOT CAMP DATA
              #   ma_final.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_games[game_ind]}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)
                 

              # DOWNLOAD FILES TO SPECIFIC LOCATION
              if 'BC' in op_games[game_ind] and op_game[4:] in count:
                op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Count/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Count/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
              elif 'BC' in op_games[game_ind] and op_game[4:] in timer:
                op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Timer/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Timer/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
              elif op_games[game_ind] == 'Single1':
                op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleR/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleR/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
              elif op_games[game_ind] == 'Single2':
                op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleL/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_SLS/SingleL/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False) 
              elif op_games[game_ind] == 'Five':
                op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Five/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Five/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)        
              elif op_games[game_ind] == 'Thirty':
                op_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Thirty/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig', index = False) 
                ma_final.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_SCA_STS/Thirty/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-MA-CLEAN.csv', encoding = 'utf-8-sig', index = False)        

print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)

print("\nFOLLOWING GAMES HAVE UNKNOWN PEAKS:", game_peaks_unknown)
