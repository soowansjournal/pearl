import pandas as pd

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
#              'Single1', 'Single2', 'Five', 'Thirty']
op_games = ['BC1']
ma_games = ['BC1']

# SELECT FILES HERE
# mmdd_p_all = ['0316_P05', '0322_P06', '0402_P07', '0403_P08', '0403_P09', '0404_P10', '0404_P11', 
#               '0406_P12', '0406_P13', '0407_P14', '0407_P15', '0407_P16', '0408_P17', '0408_P18', 
#               '0411_P19', '0412_P20', '0412_P21', '0413_P22', '0420_P23', '0420_P24', '0430_P25', '0502_P26', '0516_P27', '0601_P28']
mmdd_p_all = ['0601_P28']




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
      print(op.head(3))

      # Load MA Data
      ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/MA_' + mmdd_p + '/' + ma_file)
      print(ma.head(3))

    except FileNotFoundError:
      # if directory game file doesn't exist, go to next game
      directory_unknown.append('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file)
      continue



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
    

    # DOWNLOAD CLEANED OP BOOT CAMP DATA
    op.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv',  encoding = 'utf-8-sig') 
    # DOWNLOAD CLEANED MA BOOT CAMP DATA
    ma.to_csv(rf'/Users/soowan/Downloads/2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv', encoding = 'utf-8-sig')