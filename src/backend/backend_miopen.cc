#include "backend.hpp"
#include <iostream>
#include <assert.h>

/**/

static inline void dump_dev_prop(void * prop, int dev_id){
    char * var = getenv("VERBOSE_DEVICE");
    if(!var)
        return;
    hipDeviceProp_t *hip_prop = static_cast<hipDeviceProp_t*>(prop);
    LOG_I()<<"Device " << dev_id << ": " << hip_prop->name<<std::endl;
    LOG_I()<<"\tArch:\t" << "hip_prop->gcnArch"<<std::endl;
    LOG_I()<<"\tGMem:\t" << hip_prop->totalGlobalMem/1024/1024 << " MiB"<<std::endl;
    LOG_I()<<"\twarps:\t" << hip_prop->warpSize<<std::endl;
    LOG_I()<<"\tCUs:\t" << hip_prop->multiProcessorCount<<std::endl;
    LOG_I()<<"\tMaxClk:\t" << hip_prop->clockRate<<std::endl;
    LOG_I()<<"\tMemClk:\t" << hip_prop->memoryClockRate<<std::endl;
}

struct miopen_handle_t{
public:
    int dev_id;
    hipStream_t q;
    miopenHandle_t h;
};
#define to_miopen_handle(handle) static_cast<miopen_handle_t*>(handle)


class device_timer_hip: public device_timer_t{
public:
    device_timer_hip(){}
    virtual ~device_timer_hip(){
        reset();
    }
    virtual void reset(){
        if(event_created){
            CHECK_HIP(hipEventDestroy(start_ev));
            CHECK_HIP(hipEventDestroy(stop_ev));
            event_created = false;
        }
    }
    virtual void start(){
        reset();
        CHECK_HIP(hipEventCreate(&start_ev));
        CHECK_HIP(hipEventCreate(&stop_ev));
        event_created = true;

        CHECK_HIP(hipDeviceSynchronize());
        CHECK_HIP(hipEventRecord( start_ev, queue ));
    }
    virtual void stop(){
        CHECK_HIP(hipEventRecord( stop_ev, queue ));
        CHECK_HIP(hipEventSynchronize(stop_ev));
    }
    virtual double elapsed(){
        float ms;
        CHECK_HIP(hipEventElapsedTime(&ms,start_ev, stop_ev));
        return (double)ms;
    }

    hipStream_t queue = NULL;
    hipEvent_t start_ev, stop_ev;
    bool event_created = false;     // stupid flag to control event creation
};

device_hip::device_hip(int dev_id){
    this->type = DEVICE_HIP;
    int devcount;
    CHECK_HIP(hipGetDeviceCount(&devcount));
    assert(dev_id < devcount && "dev request must small than available ");

    for(int i=0;i<devcount;i++){
        hipDeviceProp_t hip_prop;
        CHECK_HIP(hipGetDeviceProperties(&hip_prop, i));
        dump_dev_prop(&hip_prop, i);
    }

    miopenHandle_t h;
    hipStream_t q;
    CHECK_HIP(hipSetDevice(dev_id));
    CHECK_HIP(hipStreamCreate(&q));
    CHECK_MIO(miopenCreateWithStream(&h, q));

    this->id = dev_id;
    this->queue = q;
    this->handle = h;
}
device_hip::~device_hip(){
    miopenDestroy(this->handle);
}

double device_hip::get_theoretical_gflops(tensor_data_type data_type, int is_tensor_op)
{
    // hipDeviceProp_t devProp;
    // hipGetDeviceProperties(&devProp, 0);

    // //std::cout << "Device name " << devProp.name << std::endl;
    // //std::cout << "GCN " << devProp.gcnArch << std::endl;
    // // TODO: hard code to gfx906, 60 CU
    // if(data_type == TENSOR_DT_FLOAT)
    // {
    //     if(906 == devProp.gcnArch)
    //         return 13.41 * 1000;
    //     else if(908 == devProp.gcnArch)
    //         return 39.4 * 1000;
    // }

    return 1.0;
}
device_timer_t * device_hip::device_timer_create(){
    device_timer_hip * dt = new device_timer_hip;
    dt->queue = this->queue;
    return dt;
}
void device_hip::device_timer_destroy(device_timer_t * dt){
    if(!dt)
        return ;
    delete (device_timer_hip*)dt;
}
tensor_t * device_hip::tensor_create(size_t * dims, size_t n_dim,
        tensor_data_type data_type, tensor_layout layout){

    if(n_dim == 1 && layout == TENSOR_LAYOUT_1D){
        //void* ptr;
        //CHECK_HIP(hipMalloc(&ptr, dims[0]*data_type_unit(data_type)));
        tensor_t * tensor = new tensor_t;
        tensor->dim[0] = dims[0];
        tensor->n_dims = 1;
        tensor->data_type = data_type;
        tensor->layout = layout;
        tensor->desc = nullptr;
        //tensor->mem = ptr;

        return tensor;
    }
    assert(n_dim == 4 && "current only support 4 dim tensor");
    assert(layout == TENSOR_LAYOUT_NCHW && "current only support NCHW");

    size_t n = dims[0];
    size_t c = dims[1];
    size_t h = dims[2];
    size_t w = dims[3];
    miopenTensorDescriptor_t desc;
    CHECK_MIO(miopenCreateTensorDescriptor(&desc));
    CHECK_MIO(miopenSet4dTensorDescriptor(desc, to_miopen_data_type(data_type), n, c, h, w));

    //void* ptr;
    //CHECK_HIP(hipMalloc(&ptr, n*c*h*w*data_type_unit(data_type)));

    tensor_t * tensor = new tensor_t;
    tensor->dim[0] = n;
    tensor->dim[1] = c;
    tensor->dim[2] = h;
    tensor->dim[3] = w;
    tensor->n_dims = 4;
    tensor->data_type = data_type;
    tensor->layout = layout;
    tensor->desc = desc;
    //tensor->mem = ptr;
    return tensor;
}
void device_hip::tensor_alloc(tensor_t * tensor){
    assert(tensor && !tensor->mem);
    if(tensor->n_dims==1 && tensor->layout==TENSOR_LAYOUT_1D){
        void* ptr;
        CHECK_HIP(hipMalloc(&ptr, tensor->dim[0]*data_type_unit(tensor->data_type)));
        tensor->mem = ptr;
    }else{
        void* ptr;
        CHECK_HIP(hipMalloc(&ptr,
            tensor->dim[0]*tensor->dim[1]*tensor->dim[2]*tensor->dim[3]*data_type_unit(tensor->data_type)));
        tensor->mem = ptr;
    }
}

void device_hip::tensor_copy(void *dest, void *src, size_t bytes, tensor_copy_kind copy_kind){
    if(copy_kind == TENSOR_COPY_D2H){
        tensor_t * t = (tensor_t *)src;
        CHECK_HIP(hipMemcpyDtoH(dest, t->mem, bytes));
        hipDeviceSynchronize();
    }
    else if(copy_kind == TENSOR_COPY_H2D){
        tensor_t * t = (tensor_t *)dest;
        CHECK_HIP(hipMemcpyHtoD(t->mem, src, bytes));
        hipDeviceSynchronize();
    }
    else if (copy_kind == TENSOR_COPY_D2D){
        assert(0 && "tobe implement");
    }
}
void device_hip::tensor_destroy(tensor_t * tensor){
    if(tensor->mem)
        CHECK_HIP(hipFree(tensor->mem));
    if(tensor->desc)
        CHECK_MIO(miopenDestroyTensorDescriptor((miopenTensorDescriptor_t)tensor->desc));
    delete tensor;
}
void device_hip::tensor_set(tensor_t * tensor, unsigned char v){
    assert(tensor->mem);
    CHECK_HIP(hipMemset(tensor->mem, v, tensor->bytes()));
}
pooling_desc_t * device_hip::pooling_desc_create(int * kernel, int * stride, int * padding, int n_dims,
    pooling_mode mode){
    assert(n_dims==2 && "miopen only support 2d pooling");

    miopenPoolingMode_t pool_mode;
    miopenPoolingDescriptor_t desc;

    pool_mode = to_miopen_pooling_mode(mode);
    CHECK_MIO(miopenCreatePoolingDescriptor(&desc));
    CHECK_MIO(miopenSet2dPoolingDescriptor(desc, pool_mode,
        kernel[0], kernel[1], padding[0], padding[1], stride[0], stride[1]));

    pooling_desc_t *pooling_desc = new pooling_desc_t;
    pooling_desc->mode = mode;
    pooling_desc->n_dims = 2;
    pooling_desc->kernel[0] = kernel[0];
    pooling_desc->kernel[1] = kernel[1];
    pooling_desc->stride[0] = stride[0];
    pooling_desc->stride[1] = stride[1];
    pooling_desc->padding[0] = padding[0];
    pooling_desc->padding[1] = padding[1];

    pooling_desc->desc = desc;
    return pooling_desc;
}

void device_hip::pooling_desc_destroy(pooling_desc_t * pooling_desc){
    CHECK_MIO(miopenDestroyPoolingDescriptor((miopenPoolingDescriptor_t)pooling_desc->desc));
    delete pooling_desc;
}

activation_desc_t * device_hip::activation_desc_create(activation_mode mode, float alpha){
    float beta = .0f;
    if(mode == ACTIVATION_TANH)
        beta = 1.f; // in miopen, beta * tanh(alpha * x)
    miopenActivationDescriptor_t desc;
    CHECK_MIO(miopenCreateActivationDescriptor(&desc));
    CHECK_MIO(miopenSetActivationDescriptor(desc,
            to_miopen_activation_mode(mode), (double)alpha, (double)beta, 0));

    activation_desc_t * act_desc = new activation_desc_t;
    act_desc->mode = mode;
    act_desc->alpha = alpha;
    act_desc->desc = desc;
    return  act_desc;
}
void device_hip::activation_desc_destroy(activation_desc_t * act_desc){
    CHECK_MIO(miopenDestroyActivationDescriptor((miopenActivationDescriptor_t)act_desc->desc));
    delete act_desc;
}
convolution_desc_t * device_hip::convolution_desc_create(convolution_mode mode, tensor_data_type dt,
        int * kernel, int * stride, int * padding, int * dilation, int n_dims,
        int groups, int k, int input_c, int input_h, int input_w)
{

    assert(n_dims == 2 && "current only support 2d conv");
    miopenConvolutionDescriptor_t desc;
    CHECK_MIO(miopenCreateConvolutionDescriptor(&desc));

    miopenConvolutionMode_t conv_mode = to_miopen_convolution_mode(mode);
    // we wrap the conv mode
    if(groups > 1){
        conv_mode = miopenGroupConv;
        if(groups == input_c)
            conv_mode = miopenDepthwise;
    }

    CHECK_MIO(miopenInitConvolutionDescriptor(desc, conv_mode, padding[0], padding[1],
        stride[0], stride[1], dilation[0], dilation[1]));

    if(groups > 1)
        CHECK_MIO(miopenSetConvolutionGroupCount(desc, groups));

    convolution_desc_t * conv_desc = new convolution_desc_t;
    conv_desc->mode = mode;
    conv_desc->n_dims = n_dims;

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
    conv_desc->desc = desc;

    return conv_desc;
}
void device_hip::convolution_desc_destroy(convolution_desc_t * conv_desc)
{
    CHECK_MIO(miopenDestroyConvolutionDescriptor((miopenConvolutionDescriptor_t)conv_desc->desc ));
    delete conv_desc;
}
void dump_miopen_convolution_desc(const miopenConvolutionDescriptor_t conv_desc){
    miopenConvolutionMode_t conv_mode;
    int pad_h, pad_w, u, v, dilation_h, dilation_w;
    CHECK_MIO(miopenGetConvolutionDescriptor(conv_desc, &conv_mode,
        &pad_h, &pad_w, &u, &v, &dilation_h, &dilation_w));
    std::cout<<"<conv desc> mode:"<<conv_mode<<", pad_h:"<<pad_h<<", pad_w:"<<pad_w<<
        ", u:"<<u<<", v:"<<v<<", dilation_h:"<<dilation_h<<", dilation_w:"<<dilation_w<<std::endl;
}
void dump_miopen_tensor_desc(const miopenTensorDescriptor_t tensor_desc){
    miopenDataType_t dt;
    int n,c,h,w;
    int n_stride, c_stride, h_stride, w_stride;
    CHECK_MIO(miopenGet4dTensorDescriptor(tensor_desc, &dt, &n, &c, &h, &w,
        &n_stride, &c_stride, &h_stride, &w_stride));
    std::cout<<"<tensor desc> dt:"<<dt<<", n:"<<n<<", c:"<<c<<", h:"<<h<<", w:"<<w<<
        ", n_stride:"<<n_stride<<", c_stride:"<<c_stride<<", h_stride:"<<h_stride<<", w_stride:"<<w_stride<<std::endl;
}
