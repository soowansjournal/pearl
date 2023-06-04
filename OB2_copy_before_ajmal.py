"""To Copy and Move Files to Use with Ajmal's Algorithm"""

import shutil
import os
import re

# Define the source and destination folders
# USING RAW FILES
# TIMER
source_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Raw_BC_Timer'
destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/OB2_Raw'
# # COUNT
# source_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Raw_BC_Count'
# destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/OB2_Raw'

# # USING CLEAN FILES
# # TIMER
# source_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Timer'
# destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/OB2_Clean'
# # COUNT
# source_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/Clean_BC_Count'
# destination_folder = '/Users/soowan/Documents/PEARL/Data/Data_OB2/OB2_Clean'


# Specify the letters to search for in the filenames
letters_to_search = 'MA-CLEAN'
# letters_to_search = 'OP-CLEAN'


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


print("Files copied successfully!")
