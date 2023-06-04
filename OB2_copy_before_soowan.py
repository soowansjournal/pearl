"""To Copy and Move Files to Use with Soowan's Algorithm"""

import shutil
import os
import re

count = ['Sqt', 'StLun', 'VMODip', 'HipFlex', 'HipExt', 'HipAbd', 'Kick', 'LatStep', 'BackStep',
            'StarJump', 'SeatKnExt', 'SeatHipFlex', 'SeatStarJump']
timer = ['SeatClfStr', 'Run', 'ForStep', 'CalfStr', 'TdemStnce']


# Specify the letters to search for in the filenames
for letters_to_search in count:

    # Define the source and destination folders
    source_folder = '/Users/soowan/Library/Application Support/Holland Bloorview/BBLogVisualizer/Saves/001/Logs'
    destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal/Count'

    # Get a list of files in the source folder
    file_list = os.listdir(source_folder)

    # Filter the files based on the specified letters
    filtered_files = [file_name for file_name in file_list if re.search(letters_to_search, file_name, re.IGNORECASE)]

    # Copy the filtered files to the destination folder
    for file_name in filtered_files:
        # Construct the full file paths
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)

        # Copy the file
        shutil.copy(source_file, destination_file)

    print("COUNT - Files copied successfully!")


# Specify the letters to search for in the filenames
for letters_to_search in timer:

    # Define the source and destination folders
    source_folder = '/Users/soowan/Library/Application Support/Holland Bloorview/BBLogVisualizer/Saves/001/Logs'
    destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal/Timer'

    # Get a list of files in the source folder
    file_list = os.listdir(source_folder)

    # Filter the files based on the specified letters
    filtered_files = [file_name for file_name in file_list if re.search(letters_to_search, file_name, re.IGNORECASE)]

    # Copy the filtered files to the destination folder
    for file_name in filtered_files:
        # Construct the full file paths
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)

        # Copy the file
        shutil.copy(source_file, destination_file)

    print("TIMER - Files copied successfully!")
