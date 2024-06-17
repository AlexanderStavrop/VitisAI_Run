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
destination_path="models/ResNet20_${dimension}x${dimension}_random"

# Change directory to the destination path
echo "Changing directory to $destination_path..."
cd "$destination_path"

################################################################################ 2
echo "Quantize and calibrate"
python resnet20_cifar_vai.py --data_dir "dataset/cifar10" --model_dir "./" --batch_size 16 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib

################################################################################ 3
echo "Deploy"
python resnet20_cifar_vai.py --data_dir "dataset/cifar10" --model_dir "./" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --inspect --deploy

################################################################################ 4
echo "Process the xmodel more"
/workspace/board_setup/vck190/host_cross_compiler_setup.sh

################################################################################ 5
echo "LD_LIBRARY_PATH unset and source"
unset LD_LIBRARY_PATH
source /home/vitis-ai-user/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux

################################################################################ 6
echo "Finalizing the model"
vai_c_xir -x quantize_result/CifarResNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o ./ -n resnet20CIFAR
