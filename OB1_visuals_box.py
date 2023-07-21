import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

corr_box = pd.read_excel("/Users/soowan/Documents/VSCODE/Pearl/OB1_visuals_box.xlsx")
corr_box.head()



# 1-1) Coordinate - OP vs MA - Bootle Blast
# (Correlation (of all Bootle Blast games) vs Type of Joint)
ax = sns.boxplot(x = "Joint",
            y = "Correlation",
            # hue = "Games",
            data = corr_box,
            palette = "RdBu")
# sns.stripplot(x = "Joint",
#               y = "Correlation",
#             #   hue = "Games",
#               color = 'black',
#               data = corr_box, 
#               dodge = True)
ax.set(xlabel='Type of Joint',ylabel='Pearson Correlation [r]')

# remove extra legend handles
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:6], labels[:6], title='Bootle Blast', bbox_to_anchor=(1, 1.02), loc='upper left')
ax.set_ylim(ymax = 1.05)



# # 1-1) Coordinate - OP vs MA - Boot Camp
# # (Correlation (of all Boot Camp games) vs Type of Joint)
# ax = sns.boxplot(x = "Joint",
#             y = "Correlation",
#             # hue = "Games",
#             data = corr_box,
#             palette = "RdBu")
# # sns.stripplot(x = "Joint",
# #               y = "Correlation",
# #             #   hue = "Games",
# #               color = 'black',
# #               data = corr_box, 
# #               dodge = True)
# ax.set(xlabel='Type of Joint',ylabel='Pearson Correlation [r]')

# # remove extra legend handles
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:4], labels[:4], title='Boot Camp', bbox_to_anchor=(1, 1.02), loc='upper left')
# ax.set_ylim(ymax = 1.05)



# # 1-1) Coordinate - OP vs MA - Bootle Blast
# # (Correlation (of all Bootle Blast games) vs Type of Coordinate)
# ax = sns.boxplot(x = "Coordinate",
#             y = "Correlation",
#             # hue = "Game",
#             data = corr_box,
#             palette = 'Blues')
# sns.stripplot(x = "Coordinate",
#               y = "Correlation",
#             # hue = "Game",
#               color = 'black',
#               data = corr_box, 
#               dodge = True)
# ax.set(xlabel='Coordinate',ylabel='Pearson Correlation [r]')

# # remove extra legend handles
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:6], labels[:6], title='Bootle Blast', bbox_to_anchor=(1, 1.02), loc='upper left')
# ax.set_ylim(ymax = 1.05)



# # 1-1) Coordinate - OP vs MA - Boot Camp
# # (Correlation (of all Boot Camp games) vs Type of Coordinate)
# ax = sns.boxplot(x = "Coordinate",
#             y = "Correlation",
#             # hue = "Game",
#             data = corr_box,
#             palette = 'Blues')
# sns.stripplot(x = "Coordinate",
#               y = "Correlation",
#               # hue = "Game",
#               color = 'black',
#               data = corr_box, 
#               dodge = True)
# ax.set(xlabel='Coordinate',ylabel='Pearson Correlation [r]')

# # remove extra legend handles
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[:4], labels[:4], title='Boot Camp', bbox_to_anchor=(1, 1.02), loc='upper left')
# ax.set_ylim(ymax = 1.05)