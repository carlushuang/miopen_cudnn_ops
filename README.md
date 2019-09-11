# miopen_cudnn_ops
compare cudnn/miopen ops

## build on amd rocm platform
please install rocm following [this link](https://rocm.github.io/ROCmInstall.html), and make sure miopen is installed properly
```
sh build.sh miopen
```

## build on nvidia cuda platform
please install cuda to /usr/local/cuda, as well as cudnn
```
sh build.sh cudnn
```

## compare op with op_driver
```
# after above build step, binary will result in ./build/ directory

# run alexnet 1st conv layer command:
./build/op_driver  conv -k 64 -W 227 -H 227 -c 3 -x 11 -y 11 -u 4 -v 4 -p 1 -q 1 -n 512
# run alexnet 2nd conv layer command:
./build/op_driver  conv -k 192 -W 55 -H 55 -c 64 -x 5 -y 5 -u 1 -v 1 -p 2 -q 2 -n 512

# more detail about conv parameters:
./build/op_driver  conv -h
```

