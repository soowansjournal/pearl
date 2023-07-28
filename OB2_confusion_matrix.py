import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Given TP, FN, FP, and TN values
TP = 37
FN = 14
FP = 3
TN = 0

# Calculate total count and percentages
total = TP + FN + FP + TN
percent_TP = (TP / total) * 100
percent_FN = (FN / total) * 100
percent_FP = (FP / total) * 100
percent_TN = (TN / total) * 100

# Create the confusion matrix as a 2x2 NumPy array
confusion_matrix = np.array([[TN, FP], [FN, TP]])

# Create a heatmap
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Poor Repetition", "Quality Repetition"], 
            yticklabels=["Poor Repetition", "Quality Repetition"])

# Add percentages in brackets to the heatmap
plt.annotate(f"({percent_TN:.2f}%)", xy=(0.5, 0.1), ha="center", va="center", color="black")
plt.annotate(f"({percent_FP:.2f}%)", xy=(1.5, 0.1), ha="center", va="center", color="black")
plt.annotate(f"({percent_FN:.2f}%)", xy=(0.5, 1.1), ha="center", va="center", color="black")
plt.annotate(f"({percent_TP:.2f}%)", xy=(1.5, 1.1), ha="center", va="center", color="white")

# Add labels, title, and adjust layout if needed
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix for Hip Extension [All Participants]")

# Show the plot
plt.show()


