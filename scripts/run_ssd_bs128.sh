#!/bin/sh
rm -rf  conv_log_banner.csv conv_log.csv ; sync

CONV="conv"
if [ "x$1" = "xfp16" ]; then
CONV="convfp16"
fi

./build/op_driver ${CONV} -n 128 -c 3 -H 300 -W 300 -k 64 -y 7 -x 7 -p 3 -q 3 -u 2 -v 2 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 64 -H 75 -W 75 -k 64 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 64 -H 75 -W 75 -k 128 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 128 -H 38 -W 38 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 64 -H 75 -W 75 -k 128 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 128 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 38 -W 38 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 128 -H 38 -W 38 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 38 -W 38 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 38 -W 38 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 19 -W 19 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 19 -W 19 -k 512 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 10 -W 10 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 128 -H 10 -W 10 -k 256 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 5 -W 5 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 128 -H 5 -W 5 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 3 -W 3 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 128 -H 3 -W 3 -k 256 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 38 -W 38 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 38 -W 38 -k 324 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 19 -W 19 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 19 -W 19 -k 486 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 10 -W 10 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 512 -H 10 -W 10 -k 486 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 5 -W 5 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 5 -W 5 -k 486 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 3 -W 3 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 3 -W 3 -k 324 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 1 -W 1 -k 16 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0
./build/op_driver ${CONV} -n 128 -c 256 -H 1 -W 1 -k 324 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -g 1 -t 1 -V 0

cat conv_log_banner.csv conv_log.csv > conv_ssd_bs128.csv ; sync
