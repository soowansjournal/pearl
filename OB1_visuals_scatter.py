import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches

corr_scatter = pd.read_excel("/Users/soowan/Documents/VSCODE/Pearl/OB1_visuals_scatter.xlsx")
corr_scatter.head()


# # Create a gray color palette with three shades
# gray_palette = sns.color_palette("Greys", 3)


# 1-3) Angle - OP vs MA 
# (PowerL.L.Shoulder vs Power.L.R.Shoulder | PowerR.L.Shoulder vs PowerR.R.Shoulder | Seated.L.Hip vs Seated.R.Hip)
plt.rcParams.update({'figure.figsize':(10,8), 'figure.dpi':100})
ax = sns.lmplot(x='MA', y='OP', data=corr_scatter,scatter_kws={"color": "black"},line_kws={'color': 'blue'})
ax.set(xlabel='MA Angle [deg]',ylabel='OP Angle [deg]')

red_patch = mpatches.Patch(color='black', label='Power Left: Right Shoulder Angle')
plt.legend(handles=[red_patch],loc='upper left')

plt.show()
