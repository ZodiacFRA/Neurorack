# Neurorack // Python code

## Hardware libraries

* Screen [ST7789 library](https://github.com/pimoroni/st7789-python)
* Encoder [IOE library](https://github.com/pimoroni/ioe-python)
* Control voltage [ADS1015 library](https://github.com/pimoroni/ads1015-python)


## Software troubleshooting

* Install Pytorch from source or [NVIDIA wheels](https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-8-0-now-available/72048)
* Need to rebuild libtorch (C++) [from source](https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst]
* Need to rebuild [PureData from GitHub](https://github.com/pure-data/pure-data) (apt-get install is faulty)
* Installing [latest CUDA](https://www.seeedstudio.com/blog/2020/07/29/install-cuda-11-on-jetson-nano-and-xavier-nx/) on Jetson
* Fixing [Numba and LLVM install](https://github.com/jefflgaol/Install-Packages-Jetson-ARM-Family/issues/2)
* Fixing [Librosa install](https://learninone209186366.wordpress.com/2019/07/24/how-to-install-the-librosa-library-in-jetson-nano-or-aarch64-module/) (Warning: Replace with LLVM 10)

### Currently known issues

- Problem with latest Pip (waiting on them to resolve)
https://github.com/pypa/pip/issues/9617
- Problem with latest Numpy (waiting on them)

## Useful commands

Launching python with debugging info on ARM64
```shell
python -q -X faulthandler
```




## JB

# Flash ISO from nvidia website onto a 32GB microSD card:
https://developer.nvidia.com/embedded/downloads#?tx=$product,jetson_nano
Jetson Nano Developer Kit SD Card Image - 4.6.1 - 2022/02/23

# update, install misc
sudo apt update
sudo apt install nvidia-jetpack
sudo apt-get install libatlas-base-dev gfortran
sudo apt-get install libopenblas-base libopenmpi-dev 
sudo apt install nano

# Install torch
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl -O torch-1.9.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
pip3 install numpy torch-1.9.0-cp36-cp36m-linux_aarch64.whl

# Install hardware libs
pip3 install spidev
pip3 install pimoroni-ioexpander
pip3 install ads1015
pip3 install setuptools cppy matplotlib

# Install sound libs
sudo apt install libffi-dev
pip3 install sounddevice
sudo apt install libportaudio2
sudo apt install gfortran libopenblas-dev liblapack-dev

# Librosa installation
# If problem with tbb (version too old), rename the tbb header to prevent the building system to try linking tbb:
sudo mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.h

# then install librosa, then revert the rename
sudo apt install llvm-10
export LLVM_CONFIG='/usr/bin/llvm-config-10'
pip3 install llvmlite
pip3 install pybind11
pip3 install scipy
pip3 install librosa
pip3 install smbus tqdm adafruit-circuitpython-rgb-display future

# Install image libs
sudo apt install libjpeg-dev zlib1g-dev
sudo apt-get install libfreetype6-dev
pip install Pillow
pip install -U PyYAML

# Enable spi
sudo /opt/nvidia/jetson-io/jetson-io.py
enable both spi1 and i2s4

# Add a cron at startup or manually re enter if you get "OSError: /dev/spidev0.0 does not exist"
sudo modprobe spidev

# Done

