""" To automatically rename and move raw BC files for Ajmal's Algorithm """


import pandas as pd
import shutil
import os



# CREATE FILES (DATE & GAMES)

# op_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
# ma_games = ['BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9']
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

directory_unknown = []

# Automatically Loop Through Each Participant
for mmdd_p in mmdd_p_all:
  # Automatically Loop Through Each Game
  for game_ind in range(len(op_games)):
    print(f'\n\n\n\n\n\n\n\n{op_games[game_ind]}\n\n\n\n\n\n\n\n')
    op_file = '2023' + mmdd_p[:4] + '-' + op_games[game_ind] + "-Data.csv"
    ma_file = '2023' + mmdd_p[:4] + '-' + ma_games[game_ind] + ".csv"

    # Source file path
    op_source_file = '/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file
    ma_source_file = '/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/MA_' + mmdd_p + '/' + ma_file



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

    if 'BC' in op_games[game_ind]:
      # Destination folder path
      destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Raw_BC/'
      # New name for the copied file
      op_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{op_game}-Data-OP-CLEAN.csv'
      ma_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-{ma_game}-MA-CLEAN.csv'
      # Construct the destination file path with the new name
      op_destination_file = os.path.join(destination_folder, op_new_file_name)
      ma_destination_file = os.path.join(destination_folder, ma_new_file_name)

    elif op_games[game_ind] == 'Single1':
      # Destination folder path
      destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Raw_SLS/SingleR'
      op_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-Data-OP-CLEAN.csv'
      ma_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-MA-CLEAN.csv'
      # Construct the destination file path with the new name
      op_destination_file = os.path.join(destination_folder, op_new_file_name)
      ma_destination_file = os.path.join(destination_folder, ma_new_file_name)
    
    elif op_games[game_ind] == 'Single2':
      # Destination folder path
      destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Raw_SLS/SingleL'
      op_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-Data-OP-CLEAN.csv'
      ma_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-SLS-MA-CLEAN.csv'
      # Construct the destination file path with the new name
      op_destination_file = os.path.join(destination_folder, op_new_file_name)
      ma_destination_file = os.path.join(destination_folder, ma_new_file_name)
    
    elif op_games[game_ind] == 'Five':
      # Destination folder path
      destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Raw_STS/Five'
      # New name for the copied file
      op_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-Data-OP-CLEAN.csv'
      ma_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-MA-CLEAN.csv'
      # Construct the destination file path with the new name
      op_destination_file = os.path.join(destination_folder, op_new_file_name)
      ma_destination_file = os.path.join(destination_folder, ma_new_file_name)
    
    elif op_games[game_ind] == 'Thirty':
      # Destination folder path
      destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Raw_STS/Thirty'
      # New name for the copied file
      op_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-Data-OP-CLEAN.csv'
      ma_new_file_name = rf'2023{mmdd_p[:4]}-{mmdd_p[-3:]}-BC-StS-MA-CLEAN.csv'
      # Construct the destination file path with the new name
      op_destination_file = os.path.join(destination_folder, op_new_file_name)
      ma_destination_file = os.path.join(destination_folder, ma_new_file_name)
       
        

    try:
      # Copy the file with the new name to the destination folder
      shutil.copy(op_source_file, op_destination_file)
      shutil.copy(ma_source_file, ma_destination_file)
    except:
       directory_unknown.append(op_file)
       continue

print("\nFOLLOWING FILES DO NOT EXIST:", directory_unknown)

