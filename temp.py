import os

# Set the path to the folder
folder_path = "Data/hearings txt"

# List all files in the folder
files = os.listdir(folder_path)

# Filter for files ending with '_unprocessed.txt'
unprocessed_files = [f for f in files if f.endswith('_unprocessed.txt')]

# Print the count
print(f"Number of unprocessed files: {len(unprocessed_files)}")