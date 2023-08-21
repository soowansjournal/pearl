import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

data_table = pd.read_excel("/Users/soowan/Documents/VSCODE/Pearl/Final_Defence_Visuals.xlsx")
data_table.head()



# #1 --> BAR 1-3) Max Joint Angle - OP vs MA
# # RMSE of Hip Joint Angle vs Bootle Blast Games
# plt.figure(figsize=(15,5))

# ax = sns.barplot(x='Game',y='RMSE',hue='Hip Side',data=data_table
#            ,palette='Blues'    
#            ,estimator=np.median, errorbar=('ci', 50), capsize=0.05)

# ax.set(xlabel='Bootle Blast Games',ylabel='RMSE [deg]')
# ax.set_ylim(ymin = 0)
# ax.set_title('RMSE Comparison for Hip Joint Angle in Bootle Blast Games')



# #2 --> SCATTER 1-3) Max Joint Angle - OP vs MA 
# # Correlation of Hip Joint Angle for Exercises
# # Strength: Left Hip Angle, Right Hip Angle
# # Cardio: Left Hip Angle, Right Hip Angle
# # Seated: Left Hip Angle, Right Hip Angle
# # Static: Left Hip Angle, Right Hip Angle
# plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
# ax = sns.lmplot(x='MA', y='OP', data=data_table,scatter_kws={"color": "black"},line_kws={'color': 'blue'})
# ax.set(xlabel='MA Angle [deg]',ylabel='OP Angle [deg]')

# red_patch = mpatches.Patch(color='black', label='Static: Right Hip Angle')
# plt.legend(handles=[red_patch],loc='upper left')

# plt.show()



# #3 --> BAR 1-4) Max Hand Reach - OP vs MA
# # Correlation of Max Reach vs Bootle Blast Games
# plt.figure(figsize=(15,5))

# ax = sns.barplot(x='Game',y='Corr',hue='Coordinate',data=data_table
#            ,palette='Blues'    
#            ,estimator=np.median, errorbar=('ci', 50), capsize=0.05)

# ax.set(xlabel='Bootle Blast Games',ylabel='Correlation [r]')
# ax.set_ylim(ymin = 0, ymax = 1.1)
# ax.set_title('Correlation for Targeted Hand Reach in Bootle Blast Games')

# # Move the legend outside of the chart
# legend = ax.legend(loc='upper right', bbox_to_anchor=(1.125, 1))

# # Set title for the legend
# legend.set_title('Coordinate Type')



#4 --> BAR 1-5) Max Hand Speed - OP vs MA
# Max Speed for Bootle Blast Games
plt.figure(figsize=(15,5))

ax = sns.barplot(x='Game',y='OP Speed',hue='Side',data=data_table
           ,palette='Blues'     #'cividis'
           ,estimator=np.median, errorbar=('se', 80), capsize=0.05)

# Manual error bars
x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]

ax.set(xlabel='Bootle Blast Games',ylabel='Hand Speed [m/s]')
ax.set_ylim(ymin = 0, ymax = 0.33)
ax.set_title('Max Hand Speed in Bootle Blast Games')

# RMSE of Max Speed for Bootle Blast Games
plt.figure(figsize=(15,5))

ax = sns.barplot(x='Game',y='RMSE_m',hue='Side',data=data_table
           ,palette='Blues'     #'cividis'
           ,estimator=np.median, errorbar=('se', 80), capsize=0.05)

# Manual error bars
x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
y_coords = [p.get_height() for p in ax.patches]
plt.errorbar(x=x_coords, y=y_coords, yerr=data_table["error"], fmt="none", c= "k")

ax.set(xlabel='Bootle Blast Games',ylabel='RMSE [m/s]')
ax.set_ylim(ymin = 0, ymax = 0.33)
ax.set_title('RMSE of Max Hand Speed in Bootle Blast Games')

