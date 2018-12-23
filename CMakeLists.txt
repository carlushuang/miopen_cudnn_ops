cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

project(miopen_cudnn_ops LANGUAGES CXX)

option(WITH_MIOPEN "compile with miopen, aka amd platform" ON)
option(WITH_CUDNN "compile with cudnn, aka nv platform" OFF)

list(APPEND backend_src 
            src/backend/backend.cc
            src/backend/operator.cc 
            src/backend/op_pooling.cc )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ")

if(WITH_MIOPEN)
    message(STATUS "compile on miopen, amd platform")
    set(ROCM_PATH /opt/rocm)
    set(HIP_PATH ${ROCM_PATH}/hip)

    set(CMAKE_CXX_COMPILER ${HIP_PATH}/bin/hipcc)
    add_definitions(-DWITH_MIOPEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HIP_PATH}/include -I${ROCM_PATH}/include -I${CMAKE_CURRENT_LIST_DIR}/src/backend")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --amdgpu-target=gfx900 --amdgpu-target=gfx906")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall ")

    set(LD_FLAGS "-L${ROCM_PATH}/lib -L${ROCM_PATH}/opencl/lib/x86_64 -lMIOpen -lOpenCL -lmiopengemm -lhipblas -lrocblas")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS}")

    list(APPEND backend_src
                src/backend/backend_miopen.cc
                src/backend/op_pooling_miopen.cc )

elseif(WITH_CUDNN)

else()
    message(FATAL "unknown platform.")
endif()

add_library(backend_lib STATIC ${backend_src} )

function(op_exe TARGET)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(op_exe "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET} ${op_exe_SRCS})
    target_link_libraries(${TARGET} backend_lib ${op_exe_DEPS})
endfunction()

op_exe(op_driver SRCS src/executable/op_driver.cc)
