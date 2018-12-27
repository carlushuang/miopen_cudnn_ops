#!/bin/sh
# test all op by invoke op_driver

OP_DRIVER_DIR=./build

${OP_DRIVER_DIR=}/op_driver pooling -f 0 -m 0
${OP_DRIVER_DIR=}/op_driver pooling -f 0 -m 1
${OP_DRIVER_DIR=}/op_driver pooling -f 0 -m 2
${OP_DRIVER_DIR=}/op_driver pooling -f 0 -m 2 -p 1 -k 3 -s 3
${OP_DRIVER_DIR=}/op_driver pooling -f 0 -m 3
${OP_DRIVER_DIR=}/op_driver pooling -f 0 -m 3 -p 1 -k 3 -s 3

${OP_DRIVER_DIR=}/op_driver act -f 0 -m sigmoid
${OP_DRIVER_DIR=}/op_driver act -f 0 -m relu
${OP_DRIVER_DIR=}/op_driver act -f 0 -m tanh
${OP_DRIVER_DIR=}/op_driver act -f 0 -m clipped-relu -a 0.5
${OP_DRIVER_DIR=}/op_driver act -f 0 -m clipped-relu -a 0.01
${OP_DRIVER_DIR=}/op_driver act -f 0 -m elu -a 0.4
${OP_DRIVER_DIR=}/op_driver act -f 0 -m elu -a 0.02