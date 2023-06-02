#  '''ANALYZE FILES - Automatic Boot Camp File Naming: OB1_#_analysis_bootcamp.py '''

# import pandas as pd


# # Select Game
# # op_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
# #             'StarJump', 'Run',
# #             'SeatKnExt', 'SeatHipFlex', 'SeatStarJump',
# #             'SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']
# # ma_games = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
# #             'StarJump', 'Run',
# #             'SeatKnExt', 'SeatHipFlex', 'SeatStarJump',
# #             'SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']
# op_games = ['Run']
# ma_games = ['Run']

# # SELECT FILES HERE
# # mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
# #               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
# #               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
# #               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27']
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']


# # 1) For each game
# for game_ind in range(len(op_games)):

#     # 2) For each participant
#     for mmdd_p in mmdd_p_all:
    
#         # *** For each participant rename BC#-Game ***

#         # Load bootcamp.csv file to rename
#         bootcamp = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/bootcamp.csv") 

#         # For each row 
#         # If correct participant
#         # For each column
#         # If correct Game and cell isn't empty
#         # Rename: BC#-Game

#         for i in range(len(bootcamp)):
#             if mmdd_p in bootcamp.iloc[i,0]:
#                 for j in range(len(bootcamp.columns)):
#                     if str(bootcamp.iloc[i,j]) != 'nan':
#                         if op_games[game_ind] in str(bootcamp.iloc[i,j]):
#                             op_game = bootcamp.columns[j] + '-' + op_games[game_ind]
#                             ma_game = bootcamp.columns[j] + '-' + ma_games[game_ind]
#                             # print(mmdd_p[-3:], op_game, ma_game)
#                             break
#                         else:
#                             op_game = 'NA' + '-' + op_games[game_ind]
#                             ma_game = 'NA' + '-' + op_games[game_ind]
                    




#         op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + op_game + "-Data-OP-CLEAN.csv"
#         ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + ma_game + "-MA-CLEAN.csv"

#         #print(op_file)
#         print(ma_file)



       
#     # Group Boot Camp into Four
#     # STRENGTH
#     strength = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep']
#     strength = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep']
#     # CARDIO
#     cardio = ['StarJump', 'Run']
#     cardio = ['StarJump', 'Run']
#     # SEATED
#     seated = ['SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
#     seated = ['SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
#     # STATIC
#     static = ['SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']
#     static = ['SeatClfStr', 'ForStep', 'CalfStr', 'TdemStnce']

#     if op_games[game_ind] in strength:
#         print(f'{op_games[game_ind]} = Strength')
#     elif op_games[game_ind] in cardio:
#         print(f'{op_games[game_ind]} = Cardio')
#     elif op_games[game_ind] in seated:
#         print(f'{op_games[game_ind]} = Seated')
#     elif op_games[game_ind] in static:
#         print(f'{op_games[game_ind]} = Static')




'''CLEAN FILES - Automatic Boot Camp File Naming: OB1_clean_post0316_auto.py'''



import pandas as pd


# Select Game
# op_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
#             'Pediatric', 'Single1', 'Single2', 'Five', 'Thirty']
# ma_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
#             'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
#             'Pediatric', 'Single', 'Single', 'Five', 'Thirty']
op_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
ma_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']

# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']


# 1) For each participant
for mmdd_p in mmdd_p_all:

    # 2) For each game
    for game_ind in range(len(op_games)):
    
        # *** For each participant rename BC#-Game ***

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
                                # print(mmdd_p[-3:], op_game, ma_game)
                                break
                            else:
                                op_game = op_games[game_ind] + '-' + 'NA'
                                ma_game = ma_games[game_ind] + '-' + 'NA'
                                



        op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + op_game + "-Data-OP-CLEAN.csv"
        ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + ma_game + "-MA-CLEAN.csv"

        print(op_file)
        print(ma_file)


