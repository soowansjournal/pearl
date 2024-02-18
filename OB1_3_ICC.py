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
# bootle_blast = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']
# boot_camp = ['BC_Strength', 'BC_Cardio', 'BC_Seated', 'BC_Static']
games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']


left_overall = []
right_overall = []

# For each Game
for game in games:
   if "BC" in game:
      angle_df = load_results(f'/Users/soowan/Documents/PEARL/Data/Data_OB1/3_Angle/{game}/2023-{game[3:].upper()}-angle.csv')
   else: 
      angle_df = load_results(f'/Users/soowan/Documents/PEARL/Data/Data_OB1/3_Angle/{game}/2023-{game}-angle.csv')

   left_angle = []
   right_angle = []

   if "BC" in game:
      start = 6
   else: 
      start = 5

   # For each Joint
   for i in range(start, len(angle_df.columns), 8):
      # print(angle_df.columns[i])
      df = angle_df.iloc[:,i:i+2]

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
      if 'L.' in angle_df.columns[i]:
         left_angle.append(str(icc.round(3)) + " " + f"({str(con[0].round(3))}, {str(con[1].round(3))})")
      else: 
         right_angle.append(str(icc.round(3)) + " " + f"({str(con[0].round(3))}, {str(con[1].round(3))})")
      
   # Add a row of space
   # left_angle.append(f"{game}")
   # right_angle.append(f"{game}")
   left_angle.append(f" ")
   right_angle.append(f" ")

   # Then add to overall list of all games
   left_overall = left_overall + left_angle
   right_overall = right_overall + right_angle

# Create dataframe from left and right angle lists
final_angle = pd.DataFrame({'Left': left_overall, 'Right': right_overall})
print(final_angle)

# DOWNLOAD the OVERALL angle results --> paste into data results
if "BC" in game:
   final_angle.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/BC_Angle_ICC.csv', encoding = 'utf-8-sig', index = False) 
   print("Boot Camp Angle ICC Sucess!")
else:
   final_angle.to_csv(rf'/Users/soowan/Documents/PEARL/Data/Data_OB1/BB_Angle_ICC.csv', encoding = 'utf-8-sig', index = False) 
   print("Bootle Blast Angle ICC Sucess!")
   






