cmake_minimum_required(VERSION 3.0)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE release)
endif()

project(miopen_cudnn_ops LANGUAGES CXX)
# https://stackoverflow.com/questions/10599038/can-i-skip-cmake-compiler-tests-or-avoid-error-unrecognized-option-rdynamic
# https://cmake.org/Bug/view.php?id=9985
# bug, WA for -rdynamic flag, may result in nvcc link error
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")
set(CMAKE_STATIC_LIBRARY_LINK_C_FLAGS "")
set(CMAKE_STATIC_LIBRARY_LINK_CXX_FLAGS "")

option(WITH_MIOPEN "compile with miopen, aka amd platform" OFF)
option(WITH_CUDNN "compile with cudnn, aka nv platform" OFF)

list(APPEND backend_src
            src/backend/math.cc
            src/backend/backend.cc
            src/backend/operator.cc
            src/backend/op_convolution.cc )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_LIST_DIR}/src/backend ")

macro(add_prefix var prefix)
    # var should be a string with space seperated items
    separate_arguments(tmp_list UNIX_COMMAND ${${var}})
    set(tmp_str "")
    foreach(l ${tmp_list})
        set(tmp_str "${tmp_str} ${prefix}${l}")
    endforeach()
    set(${var} "${tmp_str}")
endmacro()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DOP_VERBOSE")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -DOP_CONV_SELECT")

if(WITH_MIOPEN)
    message(STATUS "compile on miopen, amd platform")

    set(CMAKE_CXX_COMPILER ${HIP_PATH}/bin/hipcc)
    add_definitions(-DWITH_MIOPEN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HIP_PATH}/include -I${ROCM_PATH}/include ")
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-target=gfx900 -amdgpu-target=gfx906 -amdgpu-target=gfx908")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

    set(LD_FLAGS "-L${ROCM_PATH}/lib -L${ROCM_PATH}/opencl/lib/x86_64 -lMIOpen ")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${LD_FLAGS}")

    list(APPEND backend_src
                src/backend/backend_miopen.cc
                src/backend/op_convolution_miopen.cc )

elseif(WITH_CUDNN)
    message(STATUS "compile on cudnn, nv platform")
    set(CUDA_HOME /usr/local/cuda)
    if(NOT "$ENV{CUDA_HOME}" STREQUAL "")
        set(CUDA_HOME $ENV{CUDA_HOME})
    endif()
    message(STATUS "using cuda in path:${CUDA_HOME}")
    set(CMAKE_CXX_COMPILER  ${CUDA_HOME}/bin/nvcc)
    add_definitions(-DWITH_CUDNN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -O3 -Wall -Wno-literal-suffix")

    add_prefix(CMAKE_CXX_FLAGS "-Xcompiler ")
    #message(STATUS "### CMAKE_CXX_FLAGS:${CMAKE_CXX_FLAGS}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gencode arch=compute_70,code=compute_70 \
                -gencode arch=compute_61,code=compute_61 -gencode arch=compute_60,code=compute_60")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ccbin g++ ")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lcudnn -lcublas -lcuda")
    add_prefix(CMAKE_EXE_LINKER_FLAGS "-Xlinker ")
    add_prefix(CMAKE_SHARED_LINKER_FLAGS "-Xlinker ")
    add_prefix(CMAKE_STATIC_LINKER_FLAGS "-Xlinker ")
    #message(STATUS "### LD_FLAGS:${LD_FLAGS}")

    list(APPEND backend_src
                src/backend/backend_cudnn.cc
                src/backend/op_convolution_cudnn.cc )
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
    target_link_libraries(${TARGET} backend_lib ${op_exe_DEPS} m)
endfunction()

op_exe(op_driver SRCS src/executable/op_driver.cc)
