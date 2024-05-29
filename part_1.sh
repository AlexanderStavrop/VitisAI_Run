#!/bin/bash

################################################################################ 0
# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dimension>"
    exit 1
fi

# Store the input argument in a variable
dimension=$1

################################################################################ 1
# Define the source and destination paths with the variable
source_path="models/template/"
destination_path="models/ResNet20_${dimension}x${dimension}_fake"

# Copy the template directory to the desired location
echo "Copying template directory to $destination_path..."
cp -r "$source_path" "$destination_path"
if [ $? -ne 0 ]; then
    echo "Failed to copy template directory. Exiting."
    exit 1
fi

################################################################################ 2
# Run the image generator Python script with the variable
echo "Running image generator script with dimension $dimension..."
python3 image_generator.py "$dimension"
if [ $? -ne 0 ]; then
    echo "Failed to run image generator script. Exiting."
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
