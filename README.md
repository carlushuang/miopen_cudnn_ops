# miopen_cudnn_ops
compare cudnn/miopen ops

## Build on AMD ROCm

Please install ROCm and all MIOpen components.

### ROCm
For a complete guide on how to install ROCm please refer to this guide for [Linux](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

### MIOpen

For a complete guide on how to install all MIOpen components please refer to this guide for [Linux](https://docs.amd.com/projects/MIOpen/en/latest/install.html#installing-miopen-with-pre-built-packages).

MIOpen provides an optional pre-compiled kernels package to reduce the startup latency that can be installed using `install_precompiled_kernels.sh`.

The location of the script depends on your `ROCm` version.
To search for it on the disk:

```bash
$ find / -name install_precompiled_kernels.sh
/opt/rocm-6.0.0/bin/install_precompiled_kernels.sh
```

To install the kernels simply run the script:

```bash
$ /opt/rocm-6.0.0/bin/install_precompiled_kernels.sh
using rocminfo at /opt/rocm/bin/rocminfo
sudo apt install -y miopen-hip-gfx940-228kdb
Hit:1 http://repo.radeon.com/rocm/apt/6.0 focal InRelease
Hit:2 http://security.ubuntu.com/ubuntu focal-security InRelease
Hit:3 https://repo.radeon.com/amdgpu/6.0/ubuntu focal InRelease
Hit:4 http://archive.ubuntu.com/ubuntu focal InRelease
Hit:5 http://archive.ubuntu.com/ubuntu focal-updates InRelease
Hit:6 http://archive.ubuntu.com/ubuntu focal-backports InRelease
Reading package lists... Done
Building dependency tree
Reading state information... Done
All packages are up to date.
Reading package lists... Done
Building dependency tree
Reading state information... Done
E: Unable to locate package miopen-hip-gfx940-228kdb
```

Note: It would seem that there's no pre-compiled kernels package for MI300 at this time.

### Build

```bash
$ ./build.sh miopen
-DWITH_CUDNN=OFF -DWITH_MIOPEN=ON
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- compile on miopen, amd platform
-- Configuring done (0.3s)
-- Generating done (0.0s)
-- Build files have been written to: /work/repos/miopen_cudnn_ops/build
[ 11%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/math.cc.o
[ 22%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/backend_miopen.cc.o
[ 33%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/backend.cc.o
make[2]: /opt/rocm/hip/bin/hipcc: Command not found
make[2]: /opt/rocm/hip/bin/hipcc: Command not found
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:132: CMakeFiles/backend_lib.dir/src/backend/backend_miopen.cc.o] Error 127
make[2]: *** Waiting for unfinished jobs....
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:76: CMakeFiles/backend_lib.dir/src/backend/math.cc.o] Error 127
[ 44%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/operator.cc.o
[ 55%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/op_convolution.cc.o
[ 66%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/op_convolution_miopen.cc.o
make[2]: /opt/rocm/hip/bin/hipcc: Command not found
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:90: CMakeFiles/backend_lib.dir/src/backend/backend.cc.o] Error 127
make[2]: /opt/rocm/hip/bin/hipcc: Command not found
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:104: CMakeFiles/backend_lib.dir/src/backend/operator.cc.o] Error 127
make[2]: /opt/rocm/hip/bin/hipcc: Command not found
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:118: CMakeFiles/backend_lib.dir/src/backend/op_convolution.cc.o] Error 127
make[2]: /opt/rocm/hip/bin/hipcc: Command not found
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:146: CMakeFiles/backend_lib.dir/src/backend/op_convolution_miopen.cc.o] Error 127
make[1]: *** [CMakeFiles/Makefile2:85: CMakeFiles/backend_lib.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```

By fixing line 44 in `CMakeLists.txt`:

`set(HIP_PATH ${ROCM_PATH})/hip` -> `set(HIP_PATH ${ROCM_PATH})`

```bash
$ sh build.sh miopen
-DWITH_CUDNN=OFF -DWITH_MIOPEN=ON
-- The CXX compiler identification is GNU 9.4.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- compile on miopen, amd platform
-- Configuring done (0.3s)
-- Generating done (0.0s)
-- Build files have been written to: /work/repos/miopen_cudnn_ops/build
[ 11%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/math.cc.o
[ 22%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/backend.cc.o
[ 33%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/operator.cc.o
[ 44%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/backend_miopen.cc.o
[ 66%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/op_convolution_miopen.cc.o
[ 66%] Building CXX object CMakeFiles/backend_lib.dir/src/backend/op_convolution.cc.o
clang: error: unknown argument: '-amdgpu-target=gfx900'
clang: error: unknown argument: '-amdgpu-target=gfx906'
clang: error: unknown argument: '-amdgpu-target=gfx908'
clang: error: unknown argument: '-amdgpu-target=gfx900'
clang: error: unknown argument: '-amdgpu-target=gfx906'
clang: error: unknown argument: '-amdgpu-target=gfx908'
clang: error: unknown argument: '-amdgpu-target=gfx900'
clang: error: unknown argument: '-amdgpu-target=gfx906'
clang: error: unknown argument: '-amdgpu-target=gfx908'
clang: error: unknown argument: '-amdgpu-target=gfx900'
clang: error: unknown argument: '-amdgpu-target=gfx906'
clang: error: unknown argument: '-amdgpu-target=gfx908'
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:76: CMakeFiles/backend_lib.dir/src/backend/math.cc.o] Error 1
make[2]: *** Waiting for unfinished jobs....
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:146: CMakeFiles/backend_lib.dir/src/backend/op_convolution_miopen.cc.o] Error 1
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:90: CMakeFiles/backend_lib.dir/src/backend/backend.cc.o] Error 1
clang: error: unknown argument: '-amdgpu-target=gfx900'
clang: error: unknown argument: '-amdgpu-target=gfx906'
clang: error: unknown argument: '-amdgpu-target=gfx900'
clang: error: unknown argument: '-amdgpu-target=gfx906'
clang: error: unknown argument: '-amdgpu-target=gfx908'
clang: error: unknown argument: '-amdgpu-target=gfx908'
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:118: CMakeFiles/backend_lib.dir/src/backend/op_convolution.cc.o] Error 1
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:132: CMakeFiles/backend_lib.dir/src/backend/backend_miopen.cc.o] Error 1
make[2]: *** [CMakeFiles/backend_lib.dir/build.make:104: CMakeFiles/backend_lib.dir/src/backend/operator.cc.o] Error 1
make[1]: *** [CMakeFiles/Makefile2:85: CMakeFiles/backend_lib.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```

```bash
$ apt list | grep miopen

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

miopen-hip-asan-gfx1030kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx1030kdb/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx900kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx900kdb/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx906kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx906kdb/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx908kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx908kdb/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx90akdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan-gfx90akdb/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-asan/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-dev6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-dev/focal,now 3.00.0.60000-91~20.04 amd64 [installed,automatic]
miopen-hip-gfx1030kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-gfx1030kdb/focal,now 3.00.0.60000-91~20.04 amd64 [installed]
miopen-hip-gfx900kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-gfx900kdb/focal,now 3.00.0.60000-91~20.04 amd64 [installed]
miopen-hip-gfx906kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-gfx906kdb/focal,now 3.00.0.60000-91~20.04 amd64 [installed]
miopen-hip-gfx908kdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-gfx908kdb/focal,now 3.00.0.60000-91~20.04 amd64 [installed]
miopen-hip-gfx90akdb6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip-gfx90akdb/focal,now 3.00.0.60000-91~20.04 amd64 [installed]
miopen-hip6.0.0/focal 3.00.0.60000-91~20.04 amd64
miopen-hip/focal,now 3.00.0.60000-91~20.04 amd64 [installed,automatic]
```

Note: It would seem that the pre-compiled kernels are already present but the compiler doesn't recognize the `-amdgpu-target` argument.

Note: [MIOpen Porting Guide](https://docs.amd.com/projects/MIOpen/en/latest/MIOpen_Porting_Guide.html)
- MIOpen only supports float(fp32) data-type.(?!)

Note: [MIOpen Backend Limitations](https://rocm.docs.amd.com/projects/MIOpen/en/latest/find_and_immediate.html#backend-limitations)
- Basically MIOpen HIP backend has to use rocBLAS for the datatypes it doesn't support natively.

#### CMake

`https://rocmdocs.amd.com/en/develop/conceptual/cmake-packages.html`


## Build on NVIDIA CUDA platform

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
