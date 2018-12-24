#!/bin/sh

WITH_CUDNN=`test -e $WITH_CUDNN && echo 0 || echo $WITH_CUDNN `
WITH_MIOPEN=`test -e $WITH_MIOPEN && echo 1 || echo $WITH_MIOPEN`

CMAKE_BACKEND=`test "$WITH_CUDNN"  ==  "1" && 
    echo "-DWITH_CUDNN=ON -DWITH_MIOPEN=OFF" ||
    echo "-DWITH_CUDNN=OFF -DWITH_MIOPEN=ON"`

echo $CMAKE_BACKEND

rm -rf build
mkdir build && cd build
cmake $CMAKE_BACKEND ../ || exit 1
make -j`nproc`
