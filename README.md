1. Get into VitisAI
```
cd ~/Documents/VitisAI/Vitis-AI
```

3. Start the VitisAI docker image 
```
sudo ./docker_run.sh xilinx/vitis-ai-pytorch-cpu:latest
```
4. Activate vitis-ai-pytorch
```
conda activate vitis-ai-pytorch
```

6. Change to the target directory
```
cd ResNet20
```

7. Make sure the needed files are in the target directory
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

14. Move it to the board directory
```
cp resnet20CIFAR.xmodel board/cifar10/
```

15. Zip the board directory
```
zip -r board_32x32.zip board/
```

16. Send the file over to cheetera
```
scp board_32x32.zip alex@cheetara.microlab.ntua.gr:~/Desktop/
```

17. Connect to cheetara
```
ssh alex@cheetara.microlab.ntua.gr
```

18. Send files to versal
```
scp ~/Desktop/board_32x32.zip root@192.168.1.51:~/astavropoulos/
```

19. Connect to Versal 
```
ssh root@192.168.1.51
```

20. Get into my directory
```
cd astavropoulos
```

21. Unzip the zip
```
unzip board_32x32.zip
```

22. Change to the board directory
```
cd board
```

23. make _run_src.sh_
```
chmod +x run_src.sh
```

24. Run the model
```
./run_src.sh vck190 resnet20CIFAR
```
