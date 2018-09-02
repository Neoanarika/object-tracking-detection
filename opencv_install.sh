#!/bin/sh

apt-get update  # To get the latest package lists
apt-get install build-essential cmake pkg-config
apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
apt-get install libxvidcore-dev libx264-dev
apt-get install libgtk2.0-dev
apt-get install libatlas-base-dev gfortran
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.4.2.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.4.2.zip
unzip opencv_contrib.zip
cd ~/opencv-3.4.2/ && mkdir build
cd ~/opencv-3.4.2/build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.4.2/modules \
    -D BUILD_EXAMPLES=ON ..
   
cd ~/opencv-3.4.2/build && make -j4
cd ~/opencv-3.4.2/build && make clean
cd ~/opencv-3.4.2/build && make
cd ~/opencv-3.4.2/build && sudo make install
cd ~/opencv-3.4.2/build && sudo ldconfig

conda install --channel https://conda.anaconda.org/menpo opencv3
