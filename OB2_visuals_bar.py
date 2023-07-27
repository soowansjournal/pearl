import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

corr_bar = pd.read_excel("/Users/soowan/Documents/VSCODE/Pearl/OB2_visuals_bar.xlsx")
corr_bar.head()



# 2-2) Pediatric Reach Test - OP vs MA
# (RMSE (of Left/Right Reach in XYZ) for Pediatric Reach Test)
plt.figure(figsize=(15,5))

ax = sns.barplot(x='Side',y='RMSE_m',hue='Coordinate',data=corr_bar
           ,palette='Blues'    
           ,estimator=np.median, errorbar=('ci', 50), capsize=0.05)

ax.set(xlabel='Reach Type #: Side',ylabel='RMSE [m]')
ax.set_ylim(ymin = 0)



# 2-2) Pediatric Reach Test - OP vs MA
# (Correlation (of Left/Right Reach in XYZ) for Pediatric Reach Test)
plt.figure(figsize=(15,5))

ax = sns.barplot(x='Side',y='Correlation',hue='Coordinate',data=corr_bar
           ,palette='Blues'    
           ,estimator=np.median, errorbar=('ci', 50), capsize=0.05)

ax.set(xlabel='Reach Type #: Side',ylabel='Correlation [r]')
ax.set_ylim(ymin = 0, ymax = 1.1)





