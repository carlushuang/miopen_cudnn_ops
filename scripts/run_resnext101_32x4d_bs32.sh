#!/bin/sh
rm -rf  conv_log_banner.csv conv_log.csv ; sync

CONV="conv"
if [ "x$1" = "xfp16" ]; then
CONV="convfp16"
fi

./build/op_driver ${CONV} -n 32 -c 1024 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 1024 -H 14 -W 14 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 32  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 1024 -H 14 -W 14 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 1024 -H 14 -W 14 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 1024 -H 7 -W 7 -k 1024 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 1024 -H 7 -W 7 -k 2048 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 128 -H 56 -W 56 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 128 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 2048 -H 7 -W 7 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 256 -H 28 -W 28 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 256 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 256 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 256 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 32  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 256 -H 56 -W 56 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 3 -H 224 -W 224 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 512 -H 14 -W 14 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 512 -H 14 -W 14 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 32  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 512 -H 28 -W 28 -k 1024 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 512 -H 28 -W 28 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 512 -H 28 -W 28 -k 512 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 32  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 64 -H 56 -W 56 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 32 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0


cat conv_log_banner.csv conv_log.csv > conv_resnext101_32x4d_bs32.csv ; sync