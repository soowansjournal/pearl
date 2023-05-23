import pandas as pd

# Load peaks.csv file to test
peaks = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/peaks_post0316.csv") 

# For one date(participant), one game
# For each Row in Second Column
    # If the date aligns
        # Current index to next 6 indices (total of 7)
        # index 0 - OP Peak Threshold
        # index 1 - OP Distance Between
        # index 2 - OP Index 3rd Peak
        # index 3 - OP End Frame
        # index 4 - MA Peak Threshold
        # index 5 - MA Distance Between
        # index 6 - MA Index 3rd Peak
        # index 7 - NAN

# # peaks.iloc[value, game]
# date = '0314_P02'
# game = 'Power1'
# for i in range(len(peaks)):
#     if peaks["Date_P##"][i] == date:
#         print(peaks[game][i:i+7])
#         op_thresh = int(peaks[game][i])
#         op_dist = int(peaks[game][i+1])
#         op_peak = int(peaks[game][i+2])
#         op_end = int(peaks[game][i+3])
#         ma_thresh = int(peaks[game][i+4])
#         ma_dist = int(peaks[game][i+5])
#         ma_peak = int(peaks[game][i+6])


# Loop through each date(participant), each game
# For each date
    # For each game
        # For each Row in Second Column
            # If date aligns
                # If Null Values
                    # Break to next game
                # Current index to next 6 indices (total of 7)
                # index 0 - OP Peak Threshold
                # index 1 - OP Distance Between
                # index 2 - OP Index 3rd Peak
                # index 3 - OP End Frame
                # index 4 - MA Peak Threshold
                # index 5 - MA Distance Between
                # index 6 - MA Index 3rd Peak
                # index 7 - NAN

op_games = ['Power1', 'Power2', 'Wizards', 'War', 'Jet', 'Astro', 
            'BC1', 'BC2', 'BC3', 'BC4', 'BC5', 'BC6', 'BC7', 'BC8', 'BC9',
            'Pediatric', 'Single1', 'Single2', 'Five', 'Thirty']
mmdd = '0316'  
p = 'P05'
mmdd_p = mmdd + '_' + p

game_peaks_unknown = []
for game_ind in range(len(op_games)):
    # print(op_games[game_ind])

    for i in range(len(peaks)):
        if peaks["Date_P##"][i] == mmdd_p:
            if str(peaks[op_games[game_ind]][i]) == 'nan':
                # if we dont know the peaks, go to next game
                game_peaks_unknown.append(op_games[game_ind])
                break
            else:
                # print(peaks[op_games[game_ind]][i:i+7])
                op_thresh = int(peaks[op_games[game_ind]][i])
                op_dist = int(peaks[op_games[game_ind]][i+1])
                op_peak = int(peaks[op_games[game_ind]][i+2])
                op_end = int(peaks[op_games[game_ind]][i+3])
                ma_thresh = int(peaks[op_games[game_ind]][i+4])
                ma_dist = int(peaks[op_games[game_ind]][i+5])
                ma_peak = int(peaks[op_games[game_ind]][i+6])
                print(op_games[game_ind], op_thresh, ma_peak)

print("\nPEAKS UNKNOWN FOR GAMES:", game_peaks_unknown, "\n")


