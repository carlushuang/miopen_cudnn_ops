#ifndef __BACKEND_HPP
#define __BACKEND_HPP

#include <iostream>
#include <unistd.h>
#include <assert.h>
#include <stddef.h>
#include <string>

#define OP_LOG_TO_FILE
// #define OP_CUDNN_FP16_NO_TENSORCORE

enum log_level{
    LOG_INFO = 0,
    LOG_WARNING,
    LOG_ERROR,
    LOG_FATAL,
};

std::ostream & log_to_stream(log_level level);

#define LOG_I() log_to_stream(LOG_INFO)
#define LOG_W() log_to_stream(LOG_WARNING)
#define LOG_E() log_to_stream(LOG_ERROR)
#define LOG_F() log_to_stream(LOG_FATAL)

#ifdef DEBUG
#define debug_msg(fmt, ...) printf(fmt, __VA_ARGS__)
#else
#define debug_msg(fmt, ...)
#endif

enum device_type{
    DEVICE_HIP = 0,
    DEVICE_CUDA,
    DEVICE_C,
};

enum tensor_data_type{
    TENSOR_DT_FLOAT = 0,
    TENSOR_DT_HALF,
};

static inline int data_type_unit(tensor_data_type dt){
    if(dt == TENSOR_DT_FLOAT)
        return 4;
    if(dt == TENSOR_DT_HALF)
        return 2;
    return 0;
}

static inline std::string data_type_string(tensor_data_type dt){
    if(dt == TENSOR_DT_FLOAT)
        return std::string("fp32");
    if(dt == TENSOR_DT_HALF)
#if defined(OP_CUDNN_FP16_NO_TENSORCORE) && defined(WITH_CUDNN)
        return std::string("fp16-notensorcore");
#else
        return std::string("fp16");
#endif
    return std::string("n/a");
}

enum tensor_layout{
    TENSOR_LAYOUT_INVALID = -1,
    TENSOR_LAYOUT_1D = 0,
    TENSOR_LAYOUT_NCHW,
    TENSOR_LAYOUT_NHWC,
    TENSOR_LAYOUT_NCDHW,
    TENSOR_LAYOUT_NDHWC,
};
enum tensor_copy_kind{
    TENSOR_COPY_D2H = 0,
    TENSOR_COPY_H2D,
    TENSOR_COPY_D2D,
    TENSOR_COPY_ANY
};

enum tensor_layout inline tensor_string_to_layout(std::string layout_str)
{
    if(layout_str == "NCHW")
        return TENSOR_LAYOUT_NCHW;
    if(layout_str == "NHWC")
        return TENSOR_LAYOUT_NHWC;
    if(layout_str == "NCDHW")
        return TENSOR_LAYOUT_NCDHW;
    if(layout_str == "NDHWC")
        return TENSOR_LAYOUT_NDHWC;
    assert(false);
    return TENSOR_LAYOUT_INVALID;
}

std::string inline tensor_layout_to_string(enum tensor_layout layout)
{
    if(layout == TENSOR_LAYOUT_NCHW)
        return "NCHW";
    if(layout == TENSOR_LAYOUT_NHWC)
        return "NHWC";
    if(layout == TENSOR_LAYOUT_NCDHW)
        return "NCDHW";
    if(layout == TENSOR_LAYOUT_NDHWC)
        return "NDHWC";
    assert(false);
    return "INVALID";
}

#define MAX_TENSOR_DIM 4
struct tensor_t{
    size_t dim[MAX_TENSOR_DIM] ={0};
    size_t n_dims;
    size_t bytes() const {
        size_t b=1;
        for(size_t i=0;i<n_dims;i++)
            b *= dim[i];
        return b*data_type_unit(data_type);
    }
    size_t elem() const {
        size_t e=1;
        for(size_t i=0;i<n_dims;i++)
            e *= dim[i];
        return e;
    }

    tensor_data_type data_type;
    tensor_layout    layout;
    void * desc ={nullptr};
    void * mem ={nullptr};
};

enum pooling_mode {
    POOLING_MAX = 0,
    POOLING_MAX_DETERMINISTIC,
    POOLING_AVG_EXCLUSIVE,
    POOLING_AVG_INCLUSIVE
};

enum activation_mode {
    ACTIVATION_SIGMOID = 0,
    ACTIVATION_RELU,
    ACTIVATION_TANH,
    ACTIVATION_CLIPPED_RELU,
    ACTIVATION_ELU,
    ACTIVATION_IDENTITY,
};

enum convolution_mode{
    CONVOLUTION_CONV = 0,
    CONVOLUTION_CROSS_CORRELATION,  // most nn use this mode
};

#define MAX_CONV_DIM 2
struct convolution_desc_t{
    convolution_mode mode;
    int n_dims;
    int kernel[MAX_CONV_DIM];
    int stride[MAX_CONV_DIM];
    int padding[MAX_CONV_DIM];
    int dilation[MAX_CONV_DIM];
    int groups;     // group_conv/dw_conv
    int k;          // filter number, out feature maps
    int input_c;    // input feat map
    int input_h;
    int input_w;
    void * desc;    // cudnn have convDesc/filterDesc
    void * desc_wrw;    // cudnn, wrw NHWC fp16 seems only support PSEUDO_HALF_CONFIG
};

#define MAX_POOLING_DIM 2
struct pooling_desc_t{
    pooling_mode mode;
    bool adaptive;      // adaptive avg pooling https://arxiv.org/pdf/1804.10070.pdf
    int n_dims;
    int kernel[MAX_POOLING_DIM];
    int stride[MAX_POOLING_DIM];
    int padding[MAX_POOLING_DIM];
    void * desc;
};

struct activation_desc_t{
    activation_mode mode;
    float alpha;
    void * desc;
};

class device_base;
class workspace {
public:
    workspace(device_base * dev_);
    ~workspace();
    void * get(size_t bytes, tensor_data_type dt);
    tensor_t * get_tensor(size_t bytes, tensor_data_type dt);

private:
    size_t cur_byte = {0};
    device_base * dev;
    tensor_t * workspace_tensor ={nullptr};
};

class device_timer_t{
public:
    device_timer_t(){}
    virtual ~device_timer_t(){}
    virtual void reset(){}
    virtual void start(){}
    virtual void stop(){}
    virtual double elapsed(){ return 0;}      // return in ms
};

class device_base{
public:
    device_type type;
    device_base(){ws = new workspace(this);}
    virtual ~device_base(){
        if(ws){
            LOG_E()<<"device ["<<(void*)this<<"] need call shutdown first"<<std::endl;
        }
    }
    /*
    * TODO: if destroy device without shutdown, workspace may not be able to free
    */
    virtual void shutdown(){
        delete ws;
        ws = nullptr;
    }

    virtual double get_theoretical_gflops(tensor_data_type data_type, int is_tensor_op = 0) = 0;

    virtual device_timer_t * device_timer_create() = 0;
    virtual void device_timer_destroy(device_timer_t * dt) = 0;

    // in gpu, not alloc memory at this stage. use tensor_alloc() to alloc device memory after created
    virtual tensor_t * tensor_create(size_t * dims, size_t n_dim,
                    tensor_data_type data_type, tensor_layout layout)=0;
    virtual void tensor_alloc(tensor_t * tensor) = 0;
    virtual void tensor_copy(void *dest, void *src, size_t bytes, tensor_copy_kind copy_kind)=0;
    virtual void tensor_destroy(tensor_t * tensor)=0;
    virtual void tensor_set(tensor_t * tensor, unsigned char v) = 0;

    virtual pooling_desc_t * pooling_desc_create(
        int * kernel, int * stride, int * padding, int n_dims,
        pooling_mode mode)=0;
    virtual void pooling_desc_destroy(pooling_desc_t * pooling_desc)=0;

    virtual activation_desc_t * activation_desc_create(activation_mode mode, float alpha)=0;
    virtual void activation_desc_destroy(activation_desc_t * act_desc)=0;
    virtual convolution_desc_t * convolution_desc_create(convolution_mode mode, tensor_data_type dt,
        int * kernel, int * stride, int * padding, int * dilation, int n_dims,
        int groups, int k, int input_c, int input_h, int input_w) = 0;
    virtual void convolution_desc_destroy(convolution_desc_t * conv_desc) = 0;

    workspace * ws = {nullptr};
};

#ifdef WITH_MIOPEN
#include <miopen/miopen.h>
#include <hip/hip_runtime_api.h>
#define CHECK_HIP(cmd) \
do {\
    hipError_t hip_error  = cmd;\
    if (hip_error != hipSuccess) { \
        LOG_E()<<"'"<<hipGetErrorString(hip_error)<<"'("<<hip_error<<") at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define CHECK_MIO(cmd) \
{\
    miopenStatus_t miostat = cmd;\
    if (miostat != miopenStatusSuccess) { \
        LOG_E()<<"'"<<miopenGetErrorString(miostat)<<"'("<<miostat<<") at "<<__FILE__<<":"<<__LINE__<<std::endl;\
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
            return miopenPoolingAverage;
        case POOLING_AVG_INCLUSIVE:
            LOG_W()<<"MIOpen currently implementation only support exclusive avg pooling, "
                    <<"using inclusive may result in calculation fail"<<std::endl;
            return miopenPoolingAverage;
        default:
            assert(0 && "unsupported pooling mode");
    }
}
static inline miopenActivationMode_t to_miopen_activation_mode(activation_mode mode){
    switch(mode){
        case ACTIVATION_SIGMOID:
            return miopenActivationLOGISTIC;
        case ACTIVATION_RELU:
            return miopenActivationRELU;
        case ACTIVATION_TANH:
            return miopenActivationTANH;
        case ACTIVATION_CLIPPED_RELU:
            return miopenActivationCLIPPEDRELU;
        case ACTIVATION_ELU:
            return miopenActivationELU;
        case ACTIVATION_IDENTITY:
            return miopenActivationPASTHRU;
        default:
            assert(0 && "unsupported act mode");
    }
}

static inline miopenConvolutionMode_t to_miopen_convolution_mode(convolution_mode mode){
    switch(mode){
        case CONVOLUTION_CONV:
            // LOG_E()<<"miopen only support cross correlation mode for conv"<<std::endl;
            return miopenConvolution;
        break;
        case CONVOLUTION_CROSS_CORRELATION:
            return miopenConvolution;
        break;
        default:
            assert(0 && "unsupported convolution mode");
        break;
    }
}

class device_hip: public device_base{
public:
    int id;
    hipStream_t queue;
    miopenHandle_t handle;

    device_hip(int dev_id);
    ~device_hip();

    virtual double get_theoretical_gflops(tensor_data_type data_type, int is_tensor_op = 0);

    virtual device_timer_t * device_timer_create();
    virtual void device_timer_destroy(device_timer_t * dt);

    virtual tensor_t * tensor_create(size_t * dims, size_t n_dim,
                    tensor_data_type data_type, tensor_layout layout);
    virtual void tensor_alloc(tensor_t * tensor);
    virtual void tensor_copy(void *dest, void *src, size_t bytes, tensor_copy_kind copy_kind);
    virtual void tensor_destroy(tensor_t * tensor);
    virtual void tensor_set(tensor_t * tensor, unsigned char v);

    virtual pooling_desc_t * pooling_desc_create(
        int * kernel, int * stride, int * padding, int n_dims,
        pooling_mode mode);
    virtual void pooling_desc_destroy(pooling_desc_t * pooling_desc);

    virtual activation_desc_t * activation_desc_create(activation_mode mode, float alpha);
    virtual void activation_desc_destroy(activation_desc_t * act_desc);
    virtual convolution_desc_t * convolution_desc_create(convolution_mode mode, tensor_data_type dt,
        int * kernel, int * stride, int * padding, int * dilation, int n_dims,
        int groups, int k, int input_c, int input_h, int input_w);
    virtual void convolution_desc_destroy(convolution_desc_t * conv_desc);
};
void dump_miopen_convolution_desc(const miopenConvolutionDescriptor_t conv_desc);
void dump_miopen_tensor_desc(const miopenTensorDescriptor_t tensor_desc);
#endif

#ifdef WITH_CUDNN
#include <cuda_runtime.h>
#include <cuda.h>
#include <cudnn.h>

#define CHECK_CUDA(cmd) \
do {\
    cudaError_t cuda_error  = cmd;\
    if (cuda_error != cudaSuccess) { \
        LOG_E()<<"'"<<cudaGetErrorString(cuda_error)<<"'("<<cuda_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define CHECK_CU(cmd) \
do {\
    CUresult cu_error  = cmd;\
    if (cu_error != CUDA_SUCCESS) { \
        const char * p_str;               \
        cuGetErrorString(cu_error, &p_str); \
        LOG_E()<<"'"<<p_str<<"'("<<cu_error<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
        exit(EXIT_FAILURE);\
    }\
} while(0)

#define CHECK_CUDNN(cmd) \
do {\
    cudnnStatus_t stat = cmd;\
    if (stat != CUDNN_STATUS_SUCCESS) { \
        LOG_E()<<"'"<<cudnnGetErrorString(stat)<<"'("<<stat<<")"<<" at "<<__FILE__<<":"<<__LINE__<<std::endl;\
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
        LOG_W()<<"should not use TENSOR_LAYOUT_1D with cudnn"<<std::endl;
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
static inline cudnnActivationMode_t to_cudnn_activation_mode(activation_mode mode){
    switch(mode){
        case ACTIVATION_SIGMOID:
            return CUDNN_ACTIVATION_SIGMOID;
        case ACTIVATION_RELU:
            return CUDNN_ACTIVATION_RELU;
        case ACTIVATION_TANH:
            return CUDNN_ACTIVATION_TANH;
        case ACTIVATION_CLIPPED_RELU:
            return CUDNN_ACTIVATION_CLIPPED_RELU;
        case ACTIVATION_ELU:
            return CUDNN_ACTIVATION_ELU;
        case ACTIVATION_IDENTITY:
            return CUDNN_ACTIVATION_IDENTITY;
        default:
            assert(0 && "unsupported act mode");
    }
}
static inline cudnnConvolutionMode_t to_cudnn_convolution_mode(convolution_mode mode){
    switch(mode){
        case CONVOLUTION_CONV:
            return CUDNN_CONVOLUTION;
        break;
        case CONVOLUTION_CROSS_CORRELATION:
            return CUDNN_CROSS_CORRELATION;
        break;
        default:
            assert(0 && "unsupported conv mode");
        break;
    }
    return CUDNN_CROSS_CORRELATION;
}
class device_cuda : public device_base{
public:
    device_cuda(int dev_id);
    ~device_cuda();
    int id;
    cudaStream_t queue;
    cudnnHandle_t handle;
    virtual device_timer_t * device_timer_create();
    virtual void device_timer_destroy(device_timer_t * dt);

    virtual double get_theoretical_gflops(tensor_data_type data_type, int is_tensor_op = 0);

    virtual tensor_t * tensor_create(size_t * dims, size_t n_dim,
                    tensor_data_type data_type, tensor_layout layout);
	/*
    tensor_t * filter_create(size_t * dims, size_t n_dim,
                    tensor_data_type data_type, tensor_layout layout);
					*/
    virtual void tensor_alloc(tensor_t * tensor);
    virtual void tensor_copy(void *dest, void *src, size_t bytes, tensor_copy_kind copy_kind);
    virtual void tensor_destroy(tensor_t * tensor);
    virtual void tensor_set(tensor_t * tensor, unsigned char v);

    virtual pooling_desc_t * pooling_desc_create(
        int * kernel, int * stride, int * padding, int n_dims,
        pooling_mode mode);
    virtual void pooling_desc_destroy(pooling_desc_t * pooling_desc);
    virtual activation_desc_t * activation_desc_create(activation_mode mode, float alpha);
    virtual void activation_desc_destroy(activation_desc_t * act_desc);
    virtual convolution_desc_t * convolution_desc_create(convolution_mode mode, tensor_data_type dt,
        int * kernel, int * stride, int * padding, int * dilation, int n_dims,
        int groups, int k, int input_c, int input_h, int input_w);
    virtual void convolution_desc_destroy(convolution_desc_t * conv_desc);
};

/* utility */
void dump_cudnn_convolution_desc(const cudnnConvolutionDescriptor_t conv_desc);
void dump_cudnn_filter_desc(const cudnnFilterDescriptor_t filter_desc);
void dump_cudnn_tensor_desc(const cudnnTensorDescriptor_t tensor_desc);
#endif

class device_c : public device_base{
public:
    device_c();
    ~device_c();
    virtual double get_theoretical_gflops(tensor_data_type data_type, int is_tensor_op = 0) {return 0;}
    virtual device_timer_t * device_timer_create(){return nullptr;}
    virtual void device_timer_destroy(device_timer_t * dt){}
    virtual tensor_t * tensor_create(size_t * dims, size_t n_dim,
                    tensor_data_type data_type, tensor_layout layout);
    virtual void tensor_alloc(tensor_t * tensor);
    virtual void tensor_copy(void *dest, void *src, size_t bytes, tensor_copy_kind copy_kind);
    virtual void tensor_destroy(tensor_t * tensor);
    virtual void tensor_set(tensor_t * tensor, unsigned char v);
    virtual pooling_desc_t * pooling_desc_create(
        int * kernel, int * stride, int * padding, int n_dims,
        pooling_mode mode)
        {
            pooling_desc_t *pooling_desc = new pooling_desc_t;
            pooling_desc->mode = mode;
            pooling_desc->n_dims = 2;
            pooling_desc->kernel[0] = kernel[0];
            pooling_desc->kernel[1] = kernel[1];
            pooling_desc->stride[0] = stride[0];
            pooling_desc->stride[1] = stride[1];
            pooling_desc->padding[0] = padding[0];
            pooling_desc->padding[1] = padding[1];

            pooling_desc->desc = nullptr;
            return pooling_desc;
        }
    virtual void pooling_desc_destroy(pooling_desc_t * pooling_desc){
        assert(pooling_desc);
        delete pooling_desc;
    }
    virtual activation_desc_t * activation_desc_create(activation_mode mode, float alpha){
        activation_desc_t * act_desc = new activation_desc_t;
        act_desc->mode = mode;
        act_desc->alpha = alpha;
        return act_desc;
    }
    virtual void activation_desc_destroy(activation_desc_t * act_desc){
        delete act_desc;
    }
    virtual convolution_desc_t * convolution_desc_create(convolution_mode mode, tensor_data_type dt,
        int * kernel, int * stride, int * padding, int * dilation, int n_dims,
        int groups, int k, int input_c, int input_h, int input_w){
        convolution_desc_t * conv_desc = new convolution_desc_t;
        conv_desc->mode = mode;
        conv_desc->n_dims = n_dims;
        assert(n_dims <= MAX_CONV_DIM && "conv dimension not support");
        conv_desc->kernel[0] = kernel[0];
        conv_desc->kernel[1] = kernel[1];
        conv_desc->stride[0] = stride[0];
        conv_desc->stride[1] = stride[1];
        conv_desc->padding[0] = padding[0];
        conv_desc->padding[1] = padding[1];
        conv_desc->dilation[0] = dilation[0];
        conv_desc->dilation[1] = dilation[1];
        conv_desc->groups = groups;
        conv_desc->k = k;
        conv_desc->input_c = input_c;
        conv_desc->input_h = input_h;
        conv_desc->input_w = input_w;
        conv_desc->desc = nullptr;
        return conv_desc;
    }
    virtual void convolution_desc_destroy(convolution_desc_t * conv_desc){
        delete conv_desc;
    }
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
                    LOG_E()<<"compare fail, with "<<f1[i]<<" -- "<<f2[i]<<
                        " each, delta:"<< d <<", idx:"<<i<<std::endl;;
                error_cnt++;
            }
        }
    }

    return error_cnt;
}

static inline void util_b2s(size_t bytes, char * str){
	if(bytes<1024){
		sprintf(str, "%luB", bytes);
	}else if(bytes<(1024*1024)){
		double b= (double)bytes/1024.0;
		sprintf(str, "%.2fKB", b);
	}else if(bytes<(1024*1024*1024)){
		double b= (double)bytes/(1024.0*1024);
		sprintf(str, "%.2fMB", b);
	}else{
		double b= (double)bytes/(1024.0*1024*1024);
		sprintf(str, "%.2fGB", b);
	}
}

static inline std::string util_b2string(size_t bytes)
{
    char s[64];
    util_b2s(bytes, s);
    return std::string(s);
}

#endif
