#!/bin/sh

sh scripts/run_alexnet_bs512.sh         fp16 ; sleep 3
sh scripts/run_googlenet_bs128.sh       fp16 ; sleep 3
sh scripts/run_inception_v3_bs128.sh    fp16 ; sleep 3
sh scripts/run_inception_v4_bs64.sh     fp16 ; sleep 3
sh scripts/run_mask_rcnn.sh             fp16 ; sleep 3
sh scripts/run_resnext101_32x4d_bs32.sh fp16 ; sleep 3
sh scripts/run_resnext101_64x4d_bs32.sh fp16 ; sleep 3
sh scripts/run_resnext101_64x8d_bs32.sh fp16 ; sleep 3
sh scripts/run_ssd_bs128.sh             fp16 ; sleep 3
sh scripts/run_vgg16_bs128.sh           fp16 ; sleep 3
