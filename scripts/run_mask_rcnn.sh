#!/bin/sh
rm -rf  conv_log_banner.csv conv_log.csv ; sync

./build/op_driver conv -n 2 -c 1024 -H 64 -W 64 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 1024 -H 64 -W 64 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 128 -H 128 -W 128 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 128 -H 128 -W 128 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 2048 -H 32 -W 32 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 2048 -H 32 -W 32 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 128 -W 128 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 128 -W 128 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 16 -W 16 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 256 -W 256 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 256 -W 256 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 256 -W 256 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 256 -W 256 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 256 -W 256 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 256 -W 256 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 32 -W 32 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 32 -W 32 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 64 -W 64 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 64 -W 64 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 256 -H 64 -W 64 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 3 -H 1030 -W 1030 -k 64 -y 7 -x 7 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 128 -W 128 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 128 -W 128 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 128 -W 128 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 128 -W 128 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 128 -W 128 -k 256 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 128 -W 128 -k 6 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 16 -W 16 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 16 -W 16 -k 6 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 256 -W 256 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 256 -W 256 -k 6 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 32 -W 32 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 32 -W 32 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 32 -W 32 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 32 -W 32 -k 6 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 64 -W 64 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 512 -H 64 -W 64 -k 6 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 64 -H 256 -W 256 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 64 -H 256 -W 256 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 2 -c 64 -H 256 -W 256 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 400 -c 1024 -H 1 -W 1 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 400 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 400 -c 256 -H 28 -W 28 -k 256 -y 2 -x 2 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 400 -c 256 -H 28 -W 28 -k 81 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
./build/op_driver conv -n 400 -c 256 -H 7 -W 7 -k 1024 -y 7 -x 7 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -t 1 -V 0
#./build/op_driver pool -n 2 -c 256 -H 32 -W 32 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -m max -t 1 -V 0
#./build/op_driver pool -n 2 -c 64 -H 512 -W 512 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -m max -t 1 -V 0


cat conv_log_banner.csv conv_log.csv > conv_mask_rcnn.csv ; sync