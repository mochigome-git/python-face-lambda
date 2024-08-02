# Nvdia GPU DRIVER

# https://www.nvidia.com/Download/index.aspx?lang=en-us
# search nvdia product by model
```bash
wget #Download link
sudo sh "FILE_NAME"
```

# CUDA Toolkit 12.5 Downloads

## Download Installer for Linux Amazon-Linux 2023 x86_64
```bash
pip install --upgrade torch
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
sudo dnf clean all
sudo dnf -y install cuda-toolkit-12-5
# Driver
sudo dnf -y module install nvidia-driver:latest-dkms
```

## Option 2
```bash
sudo dnf install -y dkms kernel-devel kernel-modules-extra
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/amzn2023/x86_64/cuda-amzn2023.repo
sudo dnf clean expire-cache
sudo dnf -y module install nvidia-driver:latest-dkms
sudo dnf install -y cuda-toolkit
```
## Option 3
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda-repo-amzn2023-12-5-local-12.5.0_555.42.02-1.x86_64.rpm
sudo rpm -i cuda-repo-amzn2023-12-5-local-12.5.0_555.42.02-1.x86_64.rpm
sudo dnf clean all
sudo dnf -y install cuda-toolkit-12-5
pip install --upgrade torch
```

# cuDNN Downloads
## Download Installer for Linux RHEL 9 x86_64
```bash
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
sudo dnf clean all
sudo dnf -y install cudnn
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

## basic package for amazon linux
```bash
sudo yum install git 
sudo yum install openssl-devel
sudo yum install mesa-libGL
sudo yum install python3-devel 
sudo yum groupinstall "Development Tools"
sudo yum install glibc-devel
sudo yum install pkgconfig xorg-x11-server-devel libX11-devel
sudo yum install vulkan-loader
```
## Cmake
```bash
wget https://github.com/Kitware/CMake/releases/download/v3.29.3/cmake-3.29.3.tar.gz 
tar -xvzf cmake-3.29.3.tar.gz 
cd cmake-3.29.3 
./bootstrap 
sudo make 
sudo make install 
cd .. 
rm -rf cmake-3.29.3 cmake-3.29.3.tar.gz
```

## Install dlib with CUDA support
#### compile dlib from source with CUDA support enabled. 
```bash
pip3 install -v --install-option="--no" --install-option="DLIB_USE_CUDA" dlib
pip3 install face_recognition
```
