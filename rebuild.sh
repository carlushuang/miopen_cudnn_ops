#!/bin/sh
rm -rf build
mkdir build && cd build
cmake -DWITH_MIOPEN=ON ../ || exit 1
make -j`nproc`
