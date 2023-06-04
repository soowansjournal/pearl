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
    # = 1
    # If not quality rep
    # = 0 

# 1) For each count game
for game_ind in range(len(op_games)):

    participant = []
    attempt = []
    op_attempt = []
    ma_attempt = []
    op_count = []
    ma_count = []
    true_pos = []
    false_neg = []
    false_pos = []
    true_neg = []
    df_game = pd.DataFrame()

    # 2) Go through all participants
    for mmdd_p in mmdd_p_all:

        op_sum_count = []
        ma_sum_count = []
        
        # 3) Go through Ajmal's Results File
        folder_path = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal/Count'

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):

                # 4) If correct count game file
                if mmdd_p[-3:] in filename and op_games[game_ind] in filename:
                    # print(file_path)

                    # 5.1) Read/Analyze the OP results file
                    if 'OP' in filename:
                        print(filename)
                        cols_to_load = ['StartTime', 'EndTime', 'TimerStart', 'TimerEnd', 'QualityRep']
                        op = pd.read_csv(file_path, usecols=cols_to_load)

                        for i in range(1,len(op)):
                            if op['QualityRep'][i] == True:
                                count = 1
                                op_sum_count.append(count)
                            elif op['QualityRep'][i] == False: 
                                count = 0
                                op_sum_count.append(count)
                    
                    # 5.2) Read/Analyze the MA results file
                    if 'MA' in filename:
                        print(filename)
                        cols_to_load = ['StartTime', 'EndTime', 'TimerStart', 'TimerEnd', 'QualityRep']
                        ma = pd.read_csv(file_path, usecols=cols_to_load)

                        for i in range(1, len(ma)):
                            if ma['QualityRep'][i] == True:
                                count = 1
                                ma_sum_count.append(count)
                            elif ma['QualityRep'][i] == False: 
                                count = 0
                                ma_sum_count.append(count)


        # Compare for each participant
        op_total = len(op_sum_count)
        op_true = np.sum(op_sum_count)
        print(f"{op_games[game_ind]} - {mmdd_p[-3:]} - OP: {op_total}\n")
        ma_total = len(ma_sum_count)
        ma_true = np.sum(ma_sum_count)
        print(f"{ma_games[game_ind]} - {mmdd_p[-3:]} - MA: {ma_total}\n")

        participant.append(mmdd_p[-3:])
        attempt.append(len(ma_sum_count))
        op_attempt.append(op_total)
        ma_attempt.append(ma_total)
        op_count.append(op_true)
        ma_count.append(ma_true)
        
        # Calculate TP (OP:1 MA:1) FN (OP:0 MA:1) FP (OP:1 MA:0) TN (OP:0 MA:0)
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for i in range(len(ma_sum_count)):
            if op_sum_count[i] == 1 and ma_sum_count[i] == 1:
                tp = tp + 1
            elif op_sum_count[i] == 0 and ma_sum_count[i] == 1:
                fn = fn + 1
            elif op_sum_count[i] == 1 and ma_sum_count[i] == 0:
                fp = fp + 1
            elif op_sum_count[i] == 0 and ma_sum_count[i] == 0:
                tn = tn + 1
        
        true_pos.append(tp)
        false_neg.append(fn)
        false_pos.append(fp)
        true_neg.append(tn)
    

    # Create dataframe using all participants
    df_game['Participant'] = participant
    df_game['Attempt'] = attempt
    df_game['OP Attempt'] = op_attempt
    df_game['MA Attempt'] = ma_attempt
    df_game['OP True'] = op_count
    df_game['MA True'] = ma_count
    df_game['TP'] = true_pos
    df_game['FN'] = false_neg
    df_game['FP'] = false_pos
    df_game['TN'] = true_neg
    df_game = df_game.set_index('Participant')
    
    # Download Game Results to Downloads Folder
    df_game.to_csv(rf'/Users/soowan/Downloads/2023-{op_games[game_ind]}-COUNT.csv', encoding = 'utf-8-sig')

    # # Download Game Results to Specific Folder
    # df_game.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Soowan/Count/2023-{op_games[game_ind]}-COUNT.csv', encoding = 'utf-8-sig')


        
        
                        

