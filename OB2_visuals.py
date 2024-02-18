import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix



df = pd.read_csv("/Users/soowan/Documents/VSCODE/Pearl/OB2_visuals.csv")


# Data points
x = df['Sensitivity']
y = df['Specificity']
z = df['Precision']
f1 = df['F1-Score']
acc = df['Accuracy']
labels = df['Game ID']



# ROC Curve (Sensitivity vs 1 - Specificity)
plt.scatter(1 - y, x)

# Add labels to the data points
for i, label in enumerate(labels):
    plt.annotate(label, (1-y[i], x[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize = 10)

# Add axis labels and a title
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity/Recall)')
plt.title('TPR vs FPR for Exercise')
plt.ylim(top=1.1, bottom = 0)
plt.xlim(right=1.1, left = 0)

# Display the plot
plt.show()





# PRECISION-RECALL CURVE
plt.figure()
# Create the scatter plot
scatter = plt.scatter(x, z)

# Add labels to the data points
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], z[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize = 10)

# Add axis labels and a title
plt.xlabel('Sensitivity/Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall for Exercise')
plt.ylim(top=1.05, bottom = 0.55)
plt.xlim(right=1.05, left = 0.55)

# Display the plot
plt.show()





# BUBBLE CHART: SENSITIVITY VS SPECIFICITY VS PRECISION
plt.scatter(x, y, s=np.array(z)*500, alpha=0.5)

# Add labels to the data points
for i, label in enumerate(labels):
    plt.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize = 10)

# Add axis labels and a title
plt.xlabel('Sensitivity/Recall')
plt.ylabel('Specificity')
plt.title('Specificity vs Sensitivity (Bubble Size: Precision)')
plt.ylim(top=1.1, bottom = 0)
plt.xlim(right=1.1, left = 0)

# Display the plot
plt.show()






# F1-Score vs Exercise
plt.figure()
# Create the scatter plot
scatter = plt.bar(labels, f1)

# Add labels to the data points
for i, label in enumerate(labels):
    plt.annotate(label, (labels[i], f1[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize = 10)

# Add axis labels and a title
plt.xlabel('Exercise')
plt.ylabel('F1-Score')
plt.title('F1-Score for Exercise')
plt.ylim(top=1.1, bottom = 0)

# Display the plot
plt.show()






# Accuracy vs Exercise
plt.figure()
# Create the scatter plot
scatter = plt.bar(labels, acc)

# Add labels to the data points
for i, label in enumerate(labels):
    plt.annotate(label, (labels[i], acc[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize = 10)

# Add axis labels and a title
plt.xlabel('Exercise')
plt.ylabel('Accuracy [%]')
plt.title('Accuracy for Exercise')
plt.ylim(top=1.1, bottom = 0)

# Display the plot
plt.show()










