# # *** ICC EXAMPLE PINGOUIN ***

# import pandas as pd
# import pingouin as pg


# # Data is Wide-Format
# dct = {
#     'Judge A': [1, 1, 3, 6, 6, 7, 8, 9],
#     'Judge B': [2, 3, 8, 4, 5, 5, 7, 9],
#     'Judge C': [0, 3, 1, 3, 5, 6, 7, 9],
#     'Judge D': [1, 2, 4, 3, 6, 2, 9, 8],
# }
# df = pd.DataFrame(dct)
# # print(df)

# # Converted to Long-Format
# df['index'] = df.index
# df = pd.melt(df, id_vars=['index'], value_vars=list(df)[:-1])
# # print(df)

# # Compute ICC
# results = pg.intraclass_corr(data = df, targets = 'index', raters = 'variable', ratings = 'value')
# # print(results)

# # Specify which ICC
# results = results.set_index('Description')
# print(results)
# icc = results.loc['Single random raters', 'ICC']
# print(icc.round(3))




# *** ICC USING PINGOUIN ***

import pandas as pd
import pingouin as pg

def load_results(file):
  # create dataframe from uploaded csv files using pandas.read_csv()
  results = pd.read_csv(file) 
  return results

# Select Games
orient = ['P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6']


left_overall = []
right_overall = []

# For each Game
for test in orient:
    reach_df = load_results(f'/users/soowan/Documents/PEARL/Data/Data_OB2/Results_Pediatric/{test}/2023-Pediatric-reach.csv')

    left_reach = []
    right_reach = []

    # For each Joint
    for i in range(1, len(reach_df.columns), 4):
        # print(angle_df.columns[i])
        df = reach_df.iloc[:,i:i+2]

        # Data is Wide-Format
        # print(df.head(3))

        # Converted to Long-Format
        df['index'] = df.index
        df = pd.melt(df, id_vars=['index'], value_vars=list(df)[:-1])

        # Compute ICC
        results = pg.intraclass_corr(data = df, targets = 'index', raters = 'variable', ratings = 'value')

        # Specify which ICC
        results = results.set_index('Description')
        # print(results)
        icc = results.loc['Single fixed raters', 'ICC']
        # print(icc.round(3))
        # With 95% Confidence Interval
        con = results.loc['Single fixed raters', 'CI95%']
        # print(con.round(3))

        # Organize Results
        if 'L.' in reach_df.columns[i]:
            left_reach.append(str(icc.round(3)) + " " + f"({str(con[0].round(3))}, {str(con[1].round(3))})")
        else: 
            right_reach.append(str(icc.round(3)) + " " + f"({str(con[0].round(3))}, {str(con[1].round(3))})")
      
    # Add a row of space
    # left_angle.append(f"{game}")
    # right_angle.append(f"{game}")
    left_reach.append(f" ")
    right_reach.append(f" ")

    # Then add to overall list of all games
    left_overall = left_overall + left_reach
    right_overall = right_overall + right_reach

# Create dataframe from left and right angle lists
final_reach = pd.DataFrame({'Left': left_overall, 'Right': right_overall})
print(final_reach)

# DOWNLOAD the OVERALL angle results --> paste into data results
final_reach.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Pediatric/Pediatric_ICC.csv', encoding = 'utf-8-sig', index = False) 
print("Pediatric Reach Test ICC Sucess!")







