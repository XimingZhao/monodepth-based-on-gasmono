import os
import random

def generate_image_list(directory, output_file):
    """
    Generate a list of images in the directory and its subdirectories, randomly shuffled.
    The image names are converted to 04d format without the .jpg extension.
    The first and last images in each subdirectory are excluded.
    :param directory: The parent directory to search for images.
    :param output_file: The output file where the image list will be written.
    """
    image_list = []

    # Collecting image information
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            # Get all jpg files and sort them
            files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".jpg")])
            
            # Exclude the first and last images
            for filename in files[1:-1]:
                # Remove the .jpg extension and convert the name to 04d format
                image_name = "{:04d}".format(int(filename[:-4]))
                image_list.append(f"{subdir} {image_name} l")

    # Randomly shuffle the list
    random.shuffle(image_list)

    # Writing to the file
    with open(output_file, 'w') as file:
        for item in image_list:
            file.write(item + "\n")

# Example usage
directory = '../../../datasets/20231204/training'  # Replace with your directory path
output_file = './train_files.txt'  # The output file name
generate_image_list(directory, output_file)
