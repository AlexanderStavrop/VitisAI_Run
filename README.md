1. Get into VitisA
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
source $install_path/environment-setup-cortexa72-cortexa53-xilinx-linux
```

11. Finilize the model
```
vai_c_xir -x quantize_result/CifarResNet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json -o ./ -n resnet20CIFAR
```

12. Make sure there exists a _resnet20CIFAR.xmodel_ in the outer directory

13. Move it to the board directory
```
zip -r board.zip board/
```

14. Send the file over to cheetera
```
scp board.zip alex@cheetara.microlab.ntua.gr:~/Desktop/
```

15. Connect to cheetara
```
ssh alex@cheetara.microlab.ntua.gr
```

16. Send files to versal
```
scp ~/Desktop/board.zip root@192.168.1.51:~/astavropoulos/
```

15. Connect to Versal 
```
ssh alex@192.168.1.51
```
