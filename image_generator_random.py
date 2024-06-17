import os
import sys
import random
import subprocess
from PIL import Image

def create_random_png_with_size(width, height, directory, filename):
    # Create a new image with the specified size
    image = Image.new("RGB", (width, height))
    pixels = image.load()

    # Assign random colors to each pixel
    for i in range(width):
        for j in range(height):
            pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Save the image as PNG
    image.save(os.path.join(directory, filename))

def create_cifar10(classes, num_of_images, dimensions, main_directory):
    for cls in classes:
        for i in range(num_of_images):
            create_random_png_with_size(dimensions, dimensions, main_directory + cls, str(i+1) + ".png")

def create_test(classes, num_of_images, dimensions, main_directory):
    for cls in classes:
        for i in range(num_of_images):
            create_random_png_with_size(dimensions, dimensions, main_directory + cls, cls + "_" + str(i+1) + ".png")

def zip_and_remove_directory(directory):
    # Define the tar filename
    tar_filename = os.path.basename(directory.rstrip('/')) + '.tar'

    # Change to the parent directory of the target directory
    parent_dir = os.path.dirname(directory.rstrip('/'))
    target_dir = os.path.basename(directory.rstrip('/'))

    # Create a tar file from the directory
    subprocess.run(['tar', '-cvf', tar_filename, target_dir], cwd=parent_dir, check=True)

    # Remove the directory
    subprocess.run(['rm', '-rf', directory], check=True)
    print(f"Directory {directory} has been archived as {tar_filename} and removed.")

def replace_line_containing_word(file_path, target_word, replacement):
    # Open the file for reading and writing
    with open(file_path, 'r+') as file:
        # Read all lines from the file
        lines = file.readlines()

        # Iterate through each line
        for i, line in enumerate(lines):
            # Check if the target word is in the line
            if target_word in line:
                # Replace the line containing the target word with the replacement
                lines[i] = replacement + "\n"

        # Go to the beginning of the file
        file.seek(0)
        # Write the modified lines back to the file
        file.writelines(lines)
        # Truncate the file in case the new content is shorter than the old content
        file.truncate()


if __name__ == "__main__":
    dimensions = int(sys.argv[1])
    main_directory = "models/ResNet20_" + str(dimensions) + "x" + str(dimensions) + "_random/"

    if dimensions != 32:
        ## Define the new maxpooling function string
        #new_avgpool_function = "        self.avgpool_1 = nn.AvgPool2d(kernel_size={}, stride={})".format(int(dimensions/32), int(dimensions/32))
        ## Replace the line containing the target word with the new maxpooling function
        #replace_line_containing_word(main_directory + "resnet20_cifar_vai.py", "self.avgpool_1 = nn.AvgPool2d", new_avgpool_function)

        # Define the new maxpooling function string
        new_maxpool_function = "        self.maxpool = nn.MaxPool2d(kernel_size={}, stride={})".format(int(dimensions/32), int(dimensions/32))
        # Replace the line containing the target word with the new maxpooling function
        replace_line_containing_word(main_directory + "resnet20_cifar_vai.py", "self.maxpool = nn.MaxPool2d", new_maxpool_function)


        new_input_size = "  input = torch.randn([batch_size, 3, {}, {}])".format(int(dimensions), int(dimensions))
        # Replace the line containing the target word with the new maxpooling function
        replace_line_containing_word(main_directory + "resnet20_cifar_vai.py", "input = torch.randn([batch_size", new_input_size)


    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_of_images = 10

    # Creating the datasets
    create_cifar10(classes, num_of_images, dimensions, main_directory + "dataset/cifar10/val/")
    create_test(classes, num_of_images, dimensions, main_directory + "board/cifar10/test/")

    # Zip the test directory and remove it
    zip_and_remove_directory(main_directory + "board/cifar10/test/")
