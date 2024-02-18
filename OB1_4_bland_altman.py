import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# bb_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']
bb_games = ['PowerR', 'PowerL', 'Wizards', 'War', 'Jet', 'Astro']

# 1-4) reach path
reach_path = '/Users/soowan/Documents/PEARL/Data/Data_OB1/4_Reach/'

# for each game
# open "2023-GAME-stat.csv" file
# difference = OP - MA

for game in bb_games:

    df = pd.read_csv(rf"{reach_path}{game}/2023-{game}-reach.csv")
    
    

    # Find column index for OP and MA columns
    op_index = []
    for i in range(len(df.columns)):
        if 'OP' in df.columns[i]:
            op_index.append(i)    
    # print(op_index)



    op_measurements = []
    ma_measurements = []
    joint_title = []

    # Go through the OP columns
    for i in op_index:
        measurement_op = np.array(df.iloc[:,i])
        measurement_ma = np.array(df.iloc[:,i+1])

        op_measurements.append(measurement_op)
        ma_measurements.append(measurement_ma)
        joint_title.append(df.columns[i][:-4])



        # ### TO CREATE INDIVIDUAL BLAND ALTMAN PLOTS
        # # Calculate the differences and average of the measurements
        # diff = measurement_op - measurement_ma
        # mean = np.mean([measurement_op, measurement_ma], axis=0)
        # # Sample standard deviation
        # sd = np.std(diff, ddof=1)
        # # Limits of agreement
        # loa = 1.96 * sd

        # #Create the Bland-Altman plot
        # plt.scatter(mean, diff, color='black', s=50)
        # plt.axhline(np.mean(diff), color='red', linestyle='--', label = "Mean Diff")
        # plt.axhline(np.mean(diff) + loa, color='blue', linestyle='--', label='Upper LOA')
        # plt.axhline(np.mean(diff) - loa, color='blue', linestyle='--', label='Lower LOA')
        # plt.axhline(0, color='green', linestyle='--', label='Zero Line')

        # plt.xlabel('Mean of Measurements [deg]')
        # plt.ylabel('Difference [deg]')
        # plt.title(f'{game} - {df.columns[i][3:-1]} Angle')

        # # Add line labels
        # plt.text(plt.xlim()[0], np.mean(diff), ' Mean Diff ', ha='right', va='bottom', color='red')
        # plt.text(plt.xlim()[0], np.mean(diff) + loa, ' Upper LOA ', ha='right', va='bottom', color='blue')
        # plt.text(plt.xlim()[0], np.mean(diff) - loa, ' Lower LOA ', ha='right', va='top', color='blue')
        # # Add line values
        # plt.text(plt.xlim()[1], np.mean(diff), f' {np.mean(diff):.2f}', ha='left', va='center', color='red')
        # plt.text(plt.xlim()[1], np.mean(diff) + loa, f' {np.mean(diff) + loa:.2f}', ha='left', va='center', color='blue')
        # plt.text(plt.xlim()[1], np.mean(diff) - loa, f' {np.mean(diff) - loa:.2f}', ha='left', va='center', color='blue')
        # plt.text(plt.xlim()[1], 0, ' 0', ha='left', va='center', color='green')
        # plt.show()





        ### TO CREATE N by N MATRIX OF PLOTS
        # Create the figure and gridspec
        fig = plt.figure(figsize=(8, 12))
        grid = gridspec.GridSpec(4, 2, figure=fig)

        # Loop through the measurements and create Bland-Altman plots
        for j, measurement in enumerate(op_measurements):
            row = j // 2
            col = j % 2
            ax = fig.add_subplot(grid[row, col])

            measurement_op = op_measurements[j] / 100
            measurement_ma = ma_measurements[j] / 100
            diff = measurement_op - measurement_ma
            mean = np.mean([measurement_op, measurement_ma], axis=0)
            # Sample standard deviation
            sd = np.std(diff, ddof=1)
            # Limits of agreement
            loa = 1.96 * sd
           
            # Create the Bland-Altman plot
            ax.scatter(mean, diff, color='black', s=50)
            ax.axhline(np.mean(diff), color='red', linestyle='--')
            ax.axhline(np.mean(diff), color='red', linestyle='--')
            ax.axhline(np.mean(diff) + loa, color='blue', linestyle='--')
            ax.axhline(np.mean(diff) - loa, color='blue', linestyle='--')
            ax.axhline(0, color='green', linestyle='--')

            ax.set_ylim(-0.15, 0.15)  
            ax.set_xlabel('Mean [m]')
            ax.set_ylabel('Difference [m]')
            ax.set_title(f'{joint_title[j]}')
            # ax.legend()

            # Add line values
            ax.text(plt.xlim()[1], np.mean(diff), f' {np.mean(diff):.2f}', ha='left', va='center', color='red')
            ax.text(plt.xlim()[1], np.mean(diff) + loa, f' {np.mean(diff) + loa:.2f}', ha='left', va='center', color='blue')
            ax.text(plt.xlim()[1], np.mean(diff) - loa, f' {np.mean(diff) - loa:.2f}', ha='left', va='center', color='blue')
            ax.text(plt.xlim()[1], 0, ' 0', ha='left', va='center', color='green')


        # # Set the common legend outside the subplots
        # legend_labels = ['Mean Difference', 'Upper LOA', 'Lower LOA', 'Zero Line']  # Add your desired legend labels here
        # legend_elements = [Line2D([0], [0], color='red', linestyle='--', label=label) for label in legend_labels]
        # legend = plt.legend(handles=legend_elements, title='Legend', loc='center left', bbox_to_anchor=(1.05, 0.5))
        # plt.subplots_adjust(right=0.85)

        # Adjust the spacing between subplots
        fig.tight_layout()
        # Add the overall title
        fig.suptitle(f'Bland-Altman Plots ({game} - REACH)')
        # Adjust the spacing between subplots and the overall title
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        # # Show the plots
        # plt.show()
    
    # Get the path to the "Downloads" folder
    downloads_path = os.path.expanduser('~/Downloads')
    # Save the plot to the "Downloads" folder
    file_path = os.path.join(downloads_path, f'reach_{game}_bland_altman.png')
    fig.savefig(file_path)

        