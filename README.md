# miopen_cudnn_ops
compare cudnn/miopen ops

## build on amd rocm platform
please install rocm following [this link](https://rocm.github.io/ROCmInstall.html), and make sure miopen is installed properly
```
sh rebuild.sh
```

## build on nvidia cuda platform
please install cuda to /usr/local/cuda, as well as cudnn
```
WITH_CUDNN=1 sh rebuild.sh
```

## compare op with op_driver
```
# after above build step, binary will result in ./build/ directory

# run alexnet 1st conv layer command:
./op_driver  conv -k 64 -w 227 -h 227 -c 3 -x 11 -s 4 -p 1 -n 512
# run alexnet 2nd conv layer command:
./op_driver  conv -k 192 -w 55 -h 55 -c 64 -x 5 -s 1 -p 2 -n 512
```