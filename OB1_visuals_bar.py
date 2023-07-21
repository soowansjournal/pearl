import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



corr_bar = pd.read_excel("/Users/soowan/Documents/VSCODE/Pearl/OB1_visuals_bar.xlsx")
corr_bar.head()


# 1-1) Coordinate - OP vs MA 
# (Correlation (of Left/Right Elbow + Left/Right Hand) for Bootle Blast games)
plt.figure(figsize=(15,5))

ax = sns.barplot(x='Game',y='Correlation',hue='Joint Side',data=corr_bar,order=['Power Right', 'Power Left', 'Wizards', 'Paint', 'Jetpack', 'Astro']
           ,palette= 'Blues'    #'cividis'
           ,estimator=np.median, errorbar=('ci', 80), capsize=0.05)

ax.set(xlabel='Type of Game',ylabel='Pearson Correlation [r]')
ax.set_ylim(ymin = 0.7)


# # 1-2) Segment - OP 
# # (Coefficient of Variation (of Left/Right UpperArm + Left/Right Forearm) for Bootle Blast/Boot Camp games)
# plt.figure(figsize=(15,5))

# ax = sns.barplot(x='Game',y='CoV',hue='Body Segment',data=corr_bar
#            ,palette='Blues'     #'cividis'
#            ,estimator=np.median, errorbar=('ci', 80), capsize=0.05)

# ax.set(xlabel='Type of Game',ylabel='Coefficient of Variation [%]')
# ax.set_ylim(ymin = 0)



# # 1-4) Reach - OP vs MA
# # (Correlation/RMSE (of Left/Right Reach) for Bootle Blast games)
# plt.figure(figsize=(15,5))

# ax = sns.barplot(x='Game',y='RMSE_m',hue='Joint Side',data=corr_bar,order=['Power Right', 'Power Left', 'Wizards', 'Paint', 'Jetpack', 'Astro']
#            ,palette='Blues'     #'cividis'
#            ,estimator=np.median, errorbar=('ci', 50), capsize=0.05)

# ax.set(xlabel='Type of Game',ylabel='RMSE [m]')
# ax.set_ylim(ymin = 0)


# # 1-5) Speed - OP vs MA
# # (Correlation/RMSE (of Left/Right Speed) for Bootle Blast games)
# plt.figure(figsize=(15,5))

# ax = sns.barplot(x='Game',y='Correlation',hue='Side',data=corr_bar
#            ,palette='Blues'     #'cividis'
#            ,estimator=np.median, errorbar=('se', 80), capsize=0.05)

# # Manual error bars
# x_coords = [p.get_x() + 0.5*p.get_width() for p in ax.patches]
# y_coords = [p.get_height() for p in ax.patches]
# plt.errorbar(x=x_coords, y=y_coords, yerr=corr_bar["error"], fmt="none", c= "k")

# ax.set(xlabel='Type of Game',ylabel='Correlation of Speed [R]')
# ax.set_ylim(ymin = 0)