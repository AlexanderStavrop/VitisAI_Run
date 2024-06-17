#!/bin/bash

################################################################################ 0
# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dimension> <test_type>"
    echo "  <dimension>: The dimension for the images."
    echo "  <test_type>: 1 for random, 2 for pseudo_random"
    exit 1
fi

# Store the input arguments in variables
dimension=$1
test_type=$2

# Check if the test_type argument is valid
if [ "$test_type" -ne 1 ] && [ "$test_type" -ne 2 ]; then
    echo "Invalid test_type argument. Use 1 for random or 2 for pseudo_random."
    exit 1
fi

################################################################################ 1
# Removing the board directory
echo "Removing the board directory"
rm -r board

################################################################################ 2
# Unzip the zip
echo "Unziping the target zip file"

# Define the source and destination paths with the variables
if [ "$test_type" -eq 1 ]; then
    targe_zip="board_${dimension}x${dimension}_random.zip"
elif [ "$test_type" -eq 2 ]; then
    targe_zip="board_${dimension}x${dimension}_pseudo_random.zip"
fi
unzip "$targe_zip"

################################################################################ 3
# Moving inside the board directory
echo "Moving inside the board directory"
cd board

################################################################################ 4
# Make the run_src script executable
echo "Make the run_src script executable"
chmod +x run_src.sh

################################################################################ 5
# Run the model
echo "Running the model"
./run_src.sh vck190 resnet20CIFAR

############################################################################### 6
# Changing directory to print the results
echo "Chaging directory"
cd ..

############################################################################### 7
# Printing the results
echo "Printing the results"
cat ~/astavropoulos/time_measurements.txt

