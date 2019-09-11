#!/bin/sh

usage() {
	echo "Usage: $0 <miopen|cudnn>"
	exit 1
}

if [ $# != 1 ]
then
usage
fi

if [ $1 = "miopen" ]; then
CMAKE_BACKEND="-DWITH_CUDNN=OFF -DWITH_MIOPEN=ON"
elif [ $1 = "cudnn" ]; then
CMAKE_BACKEND="-DWITH_CUDNN=ON -DWITH_MIOPEN=OFF"
else
usage
fi

echo $CMAKE_BACKEND

rm -rf build
mkdir build && cd build
cmake $CMAKE_BACKEND ../ || exit 1
make -j`nproc`
