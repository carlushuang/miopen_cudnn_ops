#!/bin/sh
rm -rf  conv_log_banner.csv conv_log.csv ; sync

CONV="conv"
if [ "x$1" = "xfp16" ]; then
CONV="convfp16"
fi

./build/op_driver ${CONV} -n 512 -c 192 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 512 -c 3 -H 227 -W 227 -k 64 -y 11 -x 11 -p 0 -q 0 -u 4 -v 4 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 512 -c 384 -H 13 -W 13 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 512 -c 384 -H 13 -W 13 -k 384 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0
./build/op_driver ${CONV} -n 512 -c 64 -H 27 -W 27 -k 192 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -m conv -g 1  -t 1 -V 0

cat conv_log_banner.csv conv_log.csv > conv_alexnet_bs512.csv  ; sync