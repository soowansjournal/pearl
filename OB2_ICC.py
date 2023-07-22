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

'''
To Calculate ICC for Standardized Clinical Assessments (Pediatric Reach Test, Single Leg Stance Test L/R, Timed Up and Go Test)
'''


import pandas as pd
import pingouin as pg

   
df_ob2 = pd.read_excel(f'/Users/soowan/Documents/VSCODE/Pearl/OB2_ICC.xlsx')

left_angle = []
right_angle = []

# For each Joint
for i in range(1, len(df_ob2.columns),2):
    # print(angle_df.columns[i])
    df = df_ob2.iloc[:,i:3]

    # # Data is Wide-Format
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
    print(icc.round(3))
    # With 95% Confidence Interval
    con = results.loc['Single fixed raters', 'CI95%']
    print(con.round(3))






