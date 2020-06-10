#!/bin/sh
rm -rf  conv_log_banner.csv conv_log.csv ; sync

CONV="conv"
if [ "x$1" = "xfp16" ]; then
CONV="convfp16"
fi

./build/op_driver ${CONV} -n 128 -c 128 -H 112 -W 112 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 128 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 56 -W 56 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 3 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 14 -W 14 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 28 -W 28 -k 512 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 64 -H 112 -W 112 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 64 -H 224 -W 224 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0

cat conv_log_banner.csv conv_log.csv > conv_vgg16_bs128.csv ; sync
