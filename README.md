# cuda_class



0508 r7000 cuda-11.3
## CUDA安装
```
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
```
= Summary =


Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-11.3/
Samples:  Installed in /home/br/

Please make sure that
 -   PATH includes /usr/local/cuda-11.3/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-11.3/lib64, or, add /usr/local/cuda-11.3/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-11.3/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 465.00 is required for CUDA 11.3 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log


## TRT版本
/home/br/program/TensorRT-8.5.3.1
## cuDNN 版本

## cuBlas包含在cudaToolkit中

### cuda-python pycuda
https://nvidia.github.io/cuda-python/index.html
```
conda install -c nvidia cuda-python
conda install -c conda-forge pycuda
```

r7000上没有bert模型，huggeface打开缓慢。而且没有pycuda等下载缓慢，还需要安装transformer
TRT的学习需要再等等了
bert下好了
需要下载libtorch
