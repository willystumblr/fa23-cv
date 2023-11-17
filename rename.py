import os

# Specify the directory where your files are located
directory = "data/h3wb/images/"

# Loop through all files in the directory
for filename in os.listdir(directory):
    if filename.startswith('S6_WalkTogether_1.'):
        # Extract the random number part
        fn = filename.split('.')

        # Construct the new filename
        new_filename = f'S6_WalkTogether.{fn[1]}.{fn[2]}'

        # Create the full paths for the old and new files
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_path, new_path)

print('File renaming complete.')