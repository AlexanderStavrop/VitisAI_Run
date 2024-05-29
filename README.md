# With scripts
1. Run the first script (model directory - dataset - board test dataset - Docker)
```
./part_1.sh 32
```
2. Activate conda
```
conda activate vitis-ai-pytorch
```
3. Run the second script (Quantize and calibrate - Deploy - Process the xmodel - Unset and source the LD_LIBRARY_PATH - Finilize the model)
```
./part_2.sh 32
```
4. Exit the docker
```
Exit
```
5. Move the xmodel to the board directory and zip it.
```
./part_3.sh 64
```
6. Move inside the working directory
```
cd models/ResNet20_32x32_fake/
```
7. Send the zip to cheetera
```
scp board_32x32.zip alex@cheetara.microlab.ntua.gr:~/Desktop/
```
8. ssh to cheetara
```
ssh alex@cheetara.microlab.ntua.gr
```
9. Send the zip to Versal
```
scp ~/Desktop/board_32x32.zip root@192.168.1.87:~/astavropoulos/
```
10. ssh to Versal
```
ssh root@192.168.1.87
```
11. Unzip the zip, run the script and print the results
```
./part_4.sh
```


# By the hand
1. Get into VitisAI directory
```
cd ~/Documents/VitisAI/Vitis-AI
```
2. Create a new directory for the model
```
cp -r ~/Desktop/template/ models/ResNet20_32x32_fake
```
3. Create the cifar10 and test data
```
python3 image_generator.py 32
```
4. Start the VitisAI docker image
```
sudo ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```
5. Activate vitis-ai-pytorch
```
conda activate vitis-ai-pytorch
```
6. Change to the target directory
```
cd models/ResNet20_32x32_fake
```
7. Make sure the needed files are in the target directory
inside ResNet20 directory
- dataset/cifar10
- resnet20_cifar_vai.py
- resnet20_cifar.pkl
8. Quantize and calibrate
```
python resnet20_cifar_vai.py --data_dir "dataset/cifar10" --model_dir "./" --batch_size 16 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
```
9. Deploy
```
python resnet20_cifar_vai.py --data_dir "dataset/cifar10" --model_dir "./" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --inspect --deploy
```
10. Make sure the quantize_results folder is created
11. Process the xmodel more
```
/workspace/board_setup/vck190/host_cross_compiler_setup.sh
```
12. Unset and source the LD_LIBRARY_PATH
```
unset LD_LIBRARY_PATH
```
```
source /home/vitis-ai-user/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
```
13. Finilize the model
```
vai_c_xir -x quantize_result/CifarResNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o ./ -n resnet20CIFAR
```
14. Make sure there exists a _resnet20CIFAR.xmodel_ in the outer directory
15. Exit
```
exit
```
16. Move into the ResNet directory
```
cd models/ResNet20_32x32_fake
```
17. Move it to the board directory
```
cp resnet20CIFAR.xmodel board/cifar10/
```
18. Create a test subdir in the cifar directory
```
cp -r dataset/cifar10/val board/cifar10/test
```
19. Zip the board directory
```
zip -r board_32x32.zip board/
```
20. Send the file over to cheetera
```
scp board_32x32.zip alex@cheetara.microlab.ntua.gr:~/Desktop/
```
21. Connect to cheetara
```
ssh alex@cheetara.microlab.ntua.gr
```
22. Send files to versal
```
scp ~/Desktop/board_32x32.zip root@192.168.1.51:~/astavropoulos/
```
23. Connect to Versal 
```
ssh root@192.168.1.51
```
24. Get into my directory
```
cd astavropoulos
```
25. Unzip the zip
```
unzip board_32x32.zip
```
26. Change to the board directory
```
cd board
```
27. make _run_src.sh_
```
chmod +x run_src.sh
```
28. Run the model
```
./run_src.sh vck190 resnet20CIFAR
```
