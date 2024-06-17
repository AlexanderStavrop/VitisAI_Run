#!/bin/bash

################################################################################ 0
# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dimension>"
    echo "  <dimension>: The dimension for the images."
    exit 1
fi

# Store the input arguments in variables
dimension=$1

################################################################################ 1
# Define the source and destination paths with the variables
source_path="models/template/"
destination_path="models/ResNet20_${dimension}x${dimension}_random"
script="image_generator_random.py"

# Copy the template directory to the desired location
echo "Copying template directory to $destination_path..."
cp -r "$source_path" "$destination_path"
if [ $? -ne 0 ]; then
    echo "Failed to copy template directory. Exiting."
    exit 1
fi

################################################################################ 2
# Run the appropriate image generator Python script with the variable
echo "Running $script script with dimension $dimension..."
python3 "$script" "$dimension"
if [ $? -ne 0 ]; then
    echo "Failed to run $script script. Exiting."
    exit 1
fi

################################################################################ 3
# Run the Docker container
echo "Running Docker container..."
sudo ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
if [ $? -ne 0 ]; then
    echo "Failed to run Docker container. Exiting."
    exit 1
fi
