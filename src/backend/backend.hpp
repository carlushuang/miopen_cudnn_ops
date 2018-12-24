#ifndef __BACKEND_HPP
#define __BACKEND_HPP

#include <iostream>
#include <unistd.h>
#include <assert.h>

enum device_type{
    DEVICE_HIP,
    DEVICE_CUDA,
    DEVICE_C,
};

enum tensor_data_type{
    TENSOR_DT_FLOAT,
    TENSOR_DT_HALF,
};

static inline int data_type_unit(tensor_data_type dt){
    if(dt == TENSOR_DT_FLOAT)
        return 4;
    if(dt == TENSOR_DT_HALF)
        return 2;
    return 0;
}
enum tensor_layout{
    TENSOR_LAYOUT_1D,
    TENSOR_LAYOUT_NCHW,
    TENSOR_LAYOUT_NHWC,
};
enum tensor_copy_kind{
    TENSOR_COPY_D2H,
    TENSOR_COPY_H2D,
    TENSOR_COPY_D2D,
    TENSOR_COPY_ANY
};

#define MAX_TENSOR_DIM 4
struct tensor_t{
    int dim[MAX_TENSOR_DIM];
    int n_dims;
    int bytes() const {
        int b=1;
        for(int i=0;i<n_dims;i++)
            b *= dim[i];
        return b*data_type_unit(data_type);
    }
    int elem() const {
        int e=1;
        for(int i=0;i<n_dims;i++)
            e *= dim[i];
        return e;
    }

    tensor_data_type data_type; 
    tensor_layout    layout;
    void * desc;
    void * mem;
};

enum pooling_mode {
    POOLING_MAX,
    POOLING_MAX_DETERMINISTIC,
    POOLING_AVG_EXCLUSIVE,
    POOLING_AVG_INCLUSIVE
};

#define MAX_POOLING_DIM 3
struct pooling_desc_t{
    pooling_mode mode;
    int n_dims;
    int kernel[MAX_POOLING_DIM];
    int stride[MAX_POOLING_DIM];
    int padding[MAX_POOLING_DIM];
    void * desc;
};


class device_base{
public:
    device_type type;
    device_base(){}
    virtual ~device_base(){}

    virtual tensor_t * tensor_create(int * dims, int n_dim, 
                    tensor_data_type data_type, tensor_layout layout)=0;
    virtual void tensor_copy(void *dest, void *src, int bytes, tensor_copy_kind copy_kind)=0;
    virtual void tensor_destroy(tensor_t * tensor)=0;

    virtual pooling_desc_t * pooling_desc_create(
        int * kernel, int * stride, int * padding, int n_dims,
        pooling_mode mode){return nullptr;}
    virtual void pooling_desc_destroy(pooling_desc_t * pooling_desc){}
};
#ifdef WITH_MIOPEN
#include <miopen/miopen.h>
#include <hip/hip_runtime_api.h>
#define CHECK_HIP(cmd) \
do {\
    hipError_t hip_error  = cmd;\
    if (hip_error != hipSuccess) { \
        std::cerr<<"ERROR: '"<<hipGetErrorString(hip_error)<<"'("<<hip_error<<") at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define CHECK_MIO(cmd) \
{\
    miopenStatus_t miostat = cmd;\
    if (miostat != miopenStatusSuccess) { \
        std::cerr<<"ERROR: '"<<miopenGetErrorString(miostat)<<"'("<<miostat<<") at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
}


static inline miopenDataType_t to_miopen_data_type(tensor_data_type data_type){
    if(data_type == TENSOR_DT_FLOAT)
        return miopenFloat;
    if(data_type == TENSOR_DT_HALF)
        return miopenHalf;
    assert(0 && "unsupported data type in miopen");
    return miopenFloat; // not reached
}

static inline miopenPoolingMode_t to_miopen_pooling_mode(pooling_mode mode){
    switch (mode) {
        case POOLING_MAX:
            return miopenPoolingMax;
        case POOLING_MAX_DETERMINISTIC:
            return miopenPoolingMax;
        case POOLING_AVG_EXCLUSIVE:
            std::cerr<<"MIOpen currently implementation only support inclusive avg pooling, "
                    <<"using exclusive may result in calculation fail"<<std::endl;
            return miopenPoolingAverage;
        case POOLING_AVG_INCLUSIVE:
            return miopenPoolingAverage;
        default:
            assert(0 && "unsupported pooling mode");
    }
}
class device_hip: public device_base{
public:
    int id;
    hipStream_t queue;
    miopenHandle_t handle;

    device_hip(int dev_id);
    ~device_hip();
    virtual tensor_t * tensor_create(int * dims, int n_dim, 
                    tensor_data_type data_type, tensor_layout layout);
    virtual void tensor_copy(void *dest, void *src, int bytes, tensor_copy_kind copy_kind);
    virtual void tensor_destroy(tensor_t * tensor);

    virtual pooling_desc_t * pooling_desc_create(
        int * kernel, int * stride, int * padding, int n_dims,
        pooling_mode mode);
    virtual void pooling_desc_destroy(pooling_desc_t * pooling_desc);
};
#endif

#ifdef WITH_CUDNN
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA(cmd) \
do {\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        std::cerr<<"ERROR: '"<<cudaGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define CHECK_CUDNN(cmd) \
do {\
    cudnnStatus_t stat = cmd;\
    if (stat != CUDNN_STATUS_SUCCESS) { \
        std::cerr<<"ERROR: '"<<cudnnGetErrorString(stat)<<"'("<<stat<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

static inline cudnnDataType_t to_cudnn_data_type(tensor_data_type data_type){
    if(data_type == TENSOR_DT_FLOAT)
        return CUDNN_DATA_FLOAT;
    if(data_type == TENSOR_DT_HALF)
        return CUDNN_DATA_HALF;
    assert(0 && "unsupported data type in cudnn");
    return CUDNN_DATA_FLOAT; // not reached
}

static inline cudnnTensorFormat_t to_cudnn_layout(tensor_layout layout){
    if(layout == TENSOR_LAYOUT_1D){
        std::cerr<<"WARNING, should not use TENSOR_LAYOUT_1D with cudnn"<<std::endl;
        return CUDNN_TENSOR_NCHW;
    }
    if(layout == TENSOR_LAYOUT_NCHW)
        return CUDNN_TENSOR_NCHW;
    if(layout == TENSOR_LAYOUT_NHWC)
        return CUDNN_TENSOR_NHWC;
    assert(0 && "unsupported layout in cudnn");
    return CUDNN_TENSOR_NCHW;
}
static inline cudnnPoolingMode_t to_cudnn_pooling_mode(pooling_mode mode){
    switch (mode) {
        case POOLING_MAX:
            return CUDNN_POOLING_MAX;
        case POOLING_MAX_DETERMINISTIC:
            return CUDNN_POOLING_MAX_DETERMINISTIC;
        case POOLING_AVG_EXCLUSIVE:
            return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        case POOLING_AVG_INCLUSIVE:
            return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        default:
            assert(0 && "unsupported pooling mode");
    }
}
class device_cuda : public device_base{
public:
    device_cuda(int dev_id);
    ~device_cuda();
    int id;
    cudaStream_t queue;
    cudnnHandle_t handle;

    virtual tensor_t * tensor_create(int * dims, int n_dim, 
                    tensor_data_type data_type, tensor_layout layout);
    virtual void tensor_copy(void *dest, void *src, int bytes, tensor_copy_kind copy_kind);
    virtual void tensor_destroy(tensor_t * tensor);

    virtual pooling_desc_t * pooling_desc_create(
        int * kernel, int * stride, int * padding, int n_dims,
        pooling_mode mode);
    virtual void pooling_desc_destroy(pooling_desc_t * pooling_desc);
};
#endif

class device_c : public device_base{
public:
    device_c();
    ~device_c();
    virtual tensor_t * tensor_create(int * dims, int n_dim, 
                    tensor_data_type data_type, tensor_layout layout);
    virtual void tensor_copy(void *dest, void *src, int bytes, tensor_copy_kind copy_kind);
    virtual void tensor_destroy(tensor_t * tensor);
};


// device handle
device_base  * device_create(device_type type, int dev_id);
void           device_destroy(device_base *handle);

#define ABS(v) ( (v)>0 ? (v):-(v))
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MAX_ERROR_PRINT 10
static inline int util_compare_data(void * m1, void * m2, 
        int elem, tensor_data_type type, float delta)
{
    float *f1, *f2;
    int error_cnt = 0;
    if(type == TENSOR_DT_FLOAT){
        f1 = (float*)m1;
        f2 = (float*)m2;
        for(int i=0;i<elem;i++){
            float d = f1[i]-f2[i];
            d = ABS(d);
            if(d>delta){
                if(error_cnt < MAX_ERROR_PRINT)
                    std::cout<<"compare fail, with "<<f1[i]<<" -- "<<f2[i]<<
                        " each, delta:"<< d <<", idx:"<<i<<std::endl;;
                error_cnt++;
            }
        }
    }

    return error_cnt;
}

#endif