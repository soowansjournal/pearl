'''CLEAN FILES - Automatic Power File Naming'''



import pandas as pd


# Select Game
# op_games = ['Power1', 'Power2']
# ma_games = ['Power1', 'Power2']
op_games = ['Power1', 'Power2']
ma_games = ['Power1', 'Power2']

# SELECT FILES HERE
# mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
#               '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0221_P01', '0314_P02', '0314_P03', '0315_P04', 
              '0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
              '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
              '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']


# 1) For each participant
for mmdd_p in mmdd_p_all:

    # 2) For each game
    for game_ind in range(len(op_games)):
    




    
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
                     




        op_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + op_game + "-Data-OP-CLEAN.csv"
        ma_file = '2023' + mmdd_p[:4] + '-' + mmdd_p[-3:] + '-' + ma_game + "-MA-CLEAN.csv"

        print(op_file)
        print(ma_file)


