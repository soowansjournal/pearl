"""To analyze Boot Camp Repetition - Count - Exercises"""

import pandas as pd
import numpy as np
import os

def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file) 
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file) 
  return ma


# Select Game
# op_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
#             'StarJump', 'SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
# ma_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
#             'StarJump', 'SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
op_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
            'StarJump', 'SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
ma_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
            'StarJump', 'SeatKnExt', 'SeatHipFlex', 'SeatStarJump']


# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']

# Steps:
# 1) For each count game
# 2) For each participants
# 3) Go through Ajmal's Results File
# 4) If correct count game file
# 5) Read/Analyze the results file - OP vs MA
    # For each row
    # If quality rep
    # Subtract the timer columns
    # Sum the time
    # If not quality rep
    # Timer = 0 

# 1) For each timer game
for game_ind in range(len(op_games)):

    participant = []
    op_timer = []
    ma_timer = []
    diff_timer = []
    per_timer = []
    df_game = pd.DataFrame()

    # 2) Go through all participants
    for mmdd_p in mmdd_p_all:

        op_sum_time = []
        ma_sum_time = []
        
        # 3) Go through Ajmal's Results File
        folder_path = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal'

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):

                # 4) If correct timer game file
                if mmdd_p[-3:] in filename and op_games[game_ind] in filename:
                    # print(file_path)

                    # 5.1) Read/Analyze the OP results file
                    if 'OP' in filename:
                        print(filename)
                        cols_to_load = ['StartTime', 'EndTime', 'TimerStart', 'TimerEnd', 'QualityRep']
                        op = pd.read_csv(file_path, usecols=cols_to_load)

                        for i in range(1,len(op)):
                            if op['QualityRep'][i] == True:
                                time = op['TimerEnd'][i] - op['TimerStart'][i]
                                op_sum_time.append(time)
                            elif op['QualityRep'][i] == False: 
                                time = 0
                                op_sum_time.append(time)
                    
                    # 5.2) Read/Analyze the MA results file
                    if 'MA' in filename:
                        print(filename)
                        cols_to_load = ['StartTime', 'EndTime', 'TimerStart', 'TimerEnd', 'QualityRep']
                        ma = pd.read_csv(file_path, usecols=cols_to_load)

                        for i in range(1, len(ma)):
                            if ma['QualityRep'][i] == True:
                                time = ma['TimerEnd'][i] - ma['TimerStart'][i]
                                ma_sum_time.append(time)
                            elif ma['QualityRep'][i] == False: 
                                time = 0
                                ma_sum_time.append(time)


        # Store results for each participant
        op_time = np.sum(op_sum_time)
        print(f"{op_games[game_ind]} - {mmdd_p[-3:]} - OP: {op_time}\n")
        ma_time = np.sum(ma_sum_time)
        print(f"{ma_games[game_ind]} - {mmdd_p[-3:]} - MA: {ma_time}\n")

        participant.append(mmdd_p[-3:])
        op_timer.append(op_time)
        ma_timer.append(ma_time)
        diff_timer.append(op_time - ma_time)
        if ma_time == 0:
            per = 0
            per_timer.append(per)
        else:
            per = round(abs(op_time - ma_time) / abs(ma_time) * 100, 3)
            per_timer.append(per)
    

    # Create dataframe using all participants
    df_game['Participant'] = participant
    df_game['OP [sec]'] = op_timer
    df_game['MA [sec]'] = ma_timer
    df_game['diff [sec]'] = diff_timer
    df_game['Error [%]'] = per
    df_game = df_game.set_index('Participant')
    
    # Download Game Results to Downloads Folder
    df_game.to_csv(rf'/Users/soowan/Downloads/2023-{op_games[game_ind]}-TIMER.csv', encoding = 'utf-8-sig')

    # # Download Game Results to Specific Folder
    # df_game.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Soowan/2023-{op_games[game_ind]}-TIMER.csv', encoding = 'utf-8-sig')


        
        
                        

