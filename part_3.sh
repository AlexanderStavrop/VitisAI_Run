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
destination_path="models/ResNet20_${dimension}x${dimension}_fake"

# Copy the template directory to the desired location
echo "Changing directory to $destination_path..."
cd "$destination_path"
if [ $? -ne 0 ]; then
    echo "Failed to copy template directory. Exiting."
    exit 1
fi

################################################################################ 2
# Define the source and destination paths with the variable
source_path="~/Desktop/template/"
destination_path="models/ResNet20_${dimension}x${dimension}_fake"

# Coping the xmodel to the board directory
echo "Coping the xmodel to the board directory"
cp resnet20CIFAR.xmodel board/cifar10/
if [ $? -ne 0 ]; then
    echo "Failed to copy template directory. Exiting."
    exit 1
fi

################################################################################ 3
echo "Zipping the board directory"
zip -r board_"$dimension"x"$dimension".zip board/
