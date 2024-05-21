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
python3 image_generator.py
```
2. Start the VitisAI docker image
```
sudo ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```
3. Activate vitis-ai-pytorch
```
conda activate vitis-ai-pytorch
```
4. Change to the target directory
```
cd models/ResNet20_
```
5. Make sure the needed files are in the target directory
inside ResNet20 directory
- dataset/cifar10
- resnet20_cifar_vai.py
- resnet20_cifar.pkl
6. Quantize and calibrate
```
python resnet20_cifar_vai.py --data_dir "dataset/cifar10" --model_dir "./" --batch_size 16 --target DPUCVDX8G_ISA3_C32B6 --quant_mode calib
```
7. Deploy
```
python resnet20_cifar_vai.py --data_dir "dataset/cifar10" --model_dir "./" --batch_size 1 --target DPUCVDX8G_ISA3_C32B6 --quant_mode test --inspect --deploy
```
8. Make sure the quantize_results folder is created
9. Process the xmodel more
```
/workspace/board_setup/vck190/host_cross_compiler_setup.sh
```
10. Unset and source the LD_LIBRARY_PATH
```
unset LD_LIBRARY_PATH
```
```
source /home/vitis-ai-user/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
```
11. Finilize the model
```
vai_c_xir -x quantize_result/CifarResNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o ./ -n resnet20CIFAR
```
12. Make sure there exists a _resnet20CIFAR.xmodel_ in the outer directory
13. Exit
```
exit
```
14. Move into the ResNet directory
```
cd models/ResNet20_
```
15. Move the board folder in the ResNet directory
```
cp -r ~/Desktop/board .
```
16. Move it to the board directory
```
cp resnet20CIFAR.xmodel board/cifar10/
```
17. Create a test subdir in the cifar directory
```
cp -r dataset/cifar10/val board/cifar10/test
```
18. Move into the board/cifar directory
```
cd board/cifar10/
```
19. Make the test tar zip
```
tar -cvf test.tar test
```
20. Move back to the main directory
```
cd ../../
```
20. Zip the board directory
```
zip -r board_32x32.zip board/
```
21. Send the file over to cheetera
```
scp board_32x32.zip alex@cheetara.microlab.ntua.gr:~/Desktop/
```
22. Connect to cheetara
```
ssh alex@cheetara.microlab.ntua.gr
```
23. Send files to versal
```
scp ~/Desktop/board_32x32.zip root@192.168.1.51:~/astavropoulos/
```
24. Connect to Versal 
```
ssh root@192.168.1.51
```
25. Get into my directory
```
cd astavropoulos
```
26. Unzip the zip
```
unzip board_32x32.zip
```
27. Change to the board directory
```
cd board
```
28. make _run_src.sh_
```
chmod +x run_src.sh
```
29. Run the model
```
./run_src.sh vck190 resnet20CIFAR
```
