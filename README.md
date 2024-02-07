# miopen_cudnn_ops
compare cudnn/miopen ops

## Build on AMD ROCm

Please install ROCm and all MIOpen components.
For a complete guide on how to install ROCm please refer to this guide for
[Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).
For a complete guide on how to install all MIOpen components please refer to
this guide for [Linux](https://docs.amd.com/projects/MIOpen/en/latest/install.html#installing-miopen-with-pre-built-packages).

MIOpen provides an optional pre-compiled kernels package to reduce the startup
latency that can be installed using `install_precompiled_kernels.sh`.

```bash
$ ./build.sh miopen
```

### Build on NVIDIA CUDA

Please install CUDA to `/usr/local/cuda`, as well as `cuDNN`.

```
sh build.sh cudnn
```

## Compare op with op_driver

After the build step, the binaries will be placed in the `./build/` directory.

```
# run alexnet 1st conv layer command:
./build/op_driver  conv -k  64 -W 227 -H 227 -c  3 -x 11 -y 11 -u 4 -v 4 -p 1 -q 1 -n 512

# run alexnet 2nd conv layer command:
./build/op_driver  conv -k 192 -W  55 -H  55 -c 64 -x  5 -y  5 -u 1 -v 1 -p 2 -q 2 -n 512

# more detail about conv parameters:
./build/op_driver  conv -h
```
