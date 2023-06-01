# TEST 1: Try Except 
import pandas as pd


def load_op(op_file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  op = pd.read_csv(op_file)
  return op 


def load_ma(ma_file):
  # create dataframe from uploaded csv files using pandas.read_csv() & skip the first few rows (3) of information
  ma = pd.read_csv(ma_file, header = 3) 
  return ma


op_games = [ 'Astro', 'BC1','Pediatric', 'Single1', 'Jet']
ma_games = ['Astro', 'BC1', 'Pediatric', 'Single', 'Jet']

mmdd_p_all = ['0221_13_P01']

for mmdd_p in mmdd_p_all:
  game_peaks_unknown = []
  for game_ind in range(len(op_games)):
    op_file = '2023' + mmdd_p[:4] + '-' + op_games[game_ind] + "-Data.csv"
    ma_file = '2023' + mmdd_p[:4] + '-' + ma_games[game_ind] + ".csv"

    try:
      # Load OP Data
      op = load_op('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file)
      # print(op.head(3))

      # Load MA Data
      ma = load_ma('/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/MA_' + mmdd_p + '/' + ma_file)
      # print(ma.head(3))

    except FileNotFoundError:
      print("Directory not found:", '/Users/soowan/Documents/PEARL/Data/Data_0551/2023_' + mmdd_p + '/OP_' + mmdd_p + '/' + op_file)
      continue

    # Code to be executed if no exception occurs
    # ...
    print(op_file, "\n")
    # Rest of the loop logic
    # ...

    