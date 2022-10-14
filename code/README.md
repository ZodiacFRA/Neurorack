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



## JB / RAVE
# Use this image with ubu20.04, cuda 1.11, torch 1.12 preinstalled
https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image

# Install 
pip3 install spidev
pip3 install pimoroni-ioexpander
pip3 install ads1015

pip3 install sounddevice
sudo apt install libportaudio2
pip3 install librosa
pip3 install smbus tqdm adafruit-circuitpython-rgb-display

# Enable spi
sudo /opt/nvidia/jetson-io/jetson-io.py
enable both spi1 and i2s

# Add a cron at startup or manually re enter if you get "OSError: /dev/spidev0.0 does not exist"
sudo modprobe spidev

# Done

