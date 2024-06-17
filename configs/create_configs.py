import os
import shutil

# Define the source directory
source_dir = '/home/ai21m034/master_project/configs/train'

# Define the file names
file_names = [
    'convnextv2_small_base.json',
    'efficentnet_b4_base.json'
]

# Define the mode combinations
mode_combinations = [
    'sampling',
    'blackout',
    'albu_aug',
    'sampling_blackout',
    'sampling_albu_aug',
    'blackout_albu_aug',
    'sampling_blackout_albu_aug'
]

# Loop through each mode combination to create directories and copy files
for mode in mode_combinations:
    # Create a new directory for the current mode combination if it doesn't exist
    mode_dir = os.path.join(source_dir, mode)
    if not os.path.exists(mode_dir):
        os.makedirs(mode_dir)
        print(f"Directory '{mode_dir}' created.")

    # Loop through each file
    for file_name in file_names:
        # Create the new file name by replacing "base" with the current mode combination
        new_file_name = file_name.replace('base', mode)

        # Define the full path for the original and new files
        original_file_path = os.path.join(source_dir, file_name)
        new_file_path = os.path.join(mode_dir, new_file_name)

        # Copy and rename the file
        shutil.copyfile(original_file_path, new_file_path)
        print(f"Copied and renamed file to {new_file_path}")

# Output the completion message
print("All files have been copied and renamed into their respective mode directories.")
