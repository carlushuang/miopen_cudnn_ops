#include "backend.hpp"
#include <iostream>
#include <assert.h>

/**/

static inline void dump_dev_prop(void * prop, int dev_id){
    hipDeviceProp_t *hip_prop = static_cast<hipDeviceProp_t*>(prop);
    std::cout<<"Device " << dev_id << ": " << hip_prop->name<<std::endl;
    std::cout<<"\tArch:\t" << hip_prop->gcnArch<<std::endl;
    std::cout<<"\tGMem:\t" << hip_prop->totalGlobalMem/1024/1024 << " MiB"<<std::endl;
    std::cout<<"\twarps:\t" << hip_prop->warpSize<<std::endl;
    std::cout<<"\tCUs:\t" << hip_prop->multiProcessorCount<<std::endl;
    std::cout<<"\tMaxClk:\t" << hip_prop->clockRate<<std::endl;
    std::cout<<"\tMemClk:\t" << hip_prop->memoryClockRate<<std::endl;

}

struct miopen_handle_t{
public:
    int dev_id;
    hipStream_t q;
    miopenHandle_t h;
};
#define to_miopen_handle(handle) static_cast<miopen_handle_t*>(handle)

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

tensor_t * device_hip::tensor_create(int * dims, int n_dim, 
        tensor_data_type data_type, tensor_layout layout){

    if(n_dim == 1 && layout == TENSOR_LAYOUT_1D){
        void* ptr;
        CHECK_HIP(hipMalloc(&ptr, dims[0]*data_type_unit(data_type)));
        tensor_t * tensor = new tensor_t;
        tensor->dim[0] = dims[0];
        tensor->n_dims = 1;
        tensor->data_type = data_type;
        tensor->layout = layout;
        tensor->desc = nullptr;
        tensor->mem = ptr;

        return tensor;
    }
    assert(n_dim == 4 && "current only support 4 dim tensor");
    assert(layout == TENSOR_LAYOUT_NCHW && "current only support NCHW");

    int n = dims[0];
    int c = dims[1];
    int h = dims[2];
    int w = dims[3];
    miopenTensorDescriptor_t desc;
    CHECK_MIO(miopenCreateTensorDescriptor(&desc));
    CHECK_MIO(miopenSet4dTensorDescriptor(desc, to_miopen_data_type(data_type), n, c, h, w));

    void* ptr;
    CHECK_HIP(hipMalloc(&ptr, n*c*h*w*data_type_unit(data_type)));

    tensor_t * tensor = new tensor_t;
    tensor->dim[0] = n;
    tensor->dim[1] = c;
    tensor->dim[2] = h;
    tensor->dim[3] = w;
    tensor->n_dims = 4;
    tensor->data_type = data_type;
    tensor->layout = layout;
    tensor->desc = desc;
    tensor->mem = ptr;
    return tensor;
}

void device_hip::tensor_copy(void *dest, void *src, int bytes, tensor_copy_kind copy_kind){
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

