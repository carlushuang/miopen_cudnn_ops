#!/bin/sh
rm -rf  conv_log_banner.csv conv_log.csv ; sync

./build/op_driver conv -n 128 -c 128 -H 112 -W 112 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 128 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 256 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 512 -H 14 -W 14 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 64 -H 112 -W 112 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver conv -n 128 -c 64 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0

cat conv_log_banner.csv conv_log.csv > conv_vgg16_bs128.csv ; sync
