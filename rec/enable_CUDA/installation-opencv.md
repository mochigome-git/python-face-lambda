


## Clone the OpenCV and OpenCV contrib repositories:
```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
```
### Create a build directory:
```bash
mkdir -p opencv/build
cd opencv/build
```

### Run CMake with CUDA support:
```bash
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 ..
```

### Build and install OpenCV:
```bash
make -j$(nproc)
sudo make install
sudo ldconfig
```

### Verify the Installation:
After installation, verify that OpenCV is correctly installed with CUDA support.
```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda
```