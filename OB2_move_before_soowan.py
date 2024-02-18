"""To MOVE Files to Use with Soowan's Algorithm"""

import shutil
import os
import re




# Specify which leg!  'R' or 'L'
leg_side = 'L'
# Specify which sit! 'Five' or 'Thirty'
sit_type = 'Thirty'





single_leg_stance = ['SLS.csv']
sit_to_stand = ['StS.csv']


# Specify the letters to search for in the filenames
for letters_to_search in single_leg_stance:

    # Define the source and destination folders
    source_folder = '/Users/soowan/Library/Application Support/Holland Bloorview/BBLogVisualizer/Saves/001/Logs'

    if leg_side == 'R':
        destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal/SLS/SingleR'
    else:
        destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal/SLS/SingleL'

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
        shutil.move(source_file, destination_file)

        print("Single_Leg_Stance - Files MOVED successfully!")


# Specify the letters to search for in the filenames
for letters_to_search in sit_to_stand:

    # Define the source and destination folders
    source_folder = '/Users/soowan/Library/Application Support/Holland Bloorview/BBLogVisualizer/Saves/001/Logs'

    if sit_type == 'Five':
        destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal/STS/Five'
    else:
        destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Results_Ajmal/STS/Thirty'

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
        shutil.move(source_file, destination_file)

        print("Sit_to_Stand - Files MOVED successfully!")