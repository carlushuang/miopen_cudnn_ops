#include "backend.hpp"

static inline void dump_dev_prop(cudaDeviceProp * prop, int dev_id){
    char * var = getenv("VERBOSE_DEVICE");
    if(!var)
        return;
    LOG_I()<<"Device " << dev_id << ": " << prop->name<<std::endl;
    LOG_I()<<"\tArch:\t" << prop->major<<prop->minor<<std::endl;  // cuda compute capability
    LOG_I()<<"\tGMem:\t" << prop->totalGlobalMem/1024/1024 << " MiB"<<std::endl;
    LOG_I()<<"\twarps:\t" << prop->warpSize<<std::endl;
    LOG_I()<<"\tCUs:\t" << prop->multiProcessorCount<<std::endl;
    LOG_I()<<"\tMaxClk:\t" << prop->clockRate<<std::endl;
    LOG_I()<<"\tMemClk:\t" << prop->memoryClockRate<<std::endl;
}

device_cuda::device_cuda(int dev_id){
    this->type = DEVICE_CUDA;
    int devcount;
    CHECK_CUDA(cudaGetDeviceCount(&devcount));
    assert(dev_id < devcount && "dev request must small than available ");

    for(int i=0;i<devcount;i++){
        cudaDeviceProp cuda_prop;
        CHECK_CUDA(cudaGetDeviceProperties(&cuda_prop, i));
        dump_dev_prop(&cuda_prop, i);
    }

    cudnnHandle_t h;
    cudaStream_t q;
    CHECK_CUDA(cudaSetDevice(dev_id));
    CHECK_CUDA(cudaStreamCreate(&q));
    CHECK_CUDNN(cudnnCreate(&h));

    this->id = dev_id;
    this->queue = q;
    this->handle = h;
}
device_cuda::~device_cuda(){
    cudnnDestroy(this->handle);
}

tensor_t * device_cuda::tensor_create(int * dims, int n_dim, 
        tensor_data_type data_type, tensor_layout layout){

    if(n_dim == 1 && layout == TENSOR_LAYOUT_1D){
        void* ptr;
        CHECK_CUDA(cudaMalloc(&ptr, dims[0]*data_type_unit(data_type)));
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

    cudnnTensorDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(desc, to_cudnn_layout(layout), to_cudnn_data_type(data_type),
        dims[0], dims[1], dims[2], dims[3]));

    void* ptr;
    CHECK_CUDA(cudaMalloc(&ptr, dims[0]*dims[1]*dims[2]*dims[3]*data_type_unit(data_type)));

    tensor_t * tensor = new tensor_t;
    tensor->dim[0] = dims[0];
    tensor->dim[1] = dims[1];
    tensor->dim[2] = dims[2];
    tensor->dim[3] = dims[3];
    tensor->n_dims = 4;
    tensor->data_type = data_type;
    tensor->layout = layout;
    tensor->desc = desc;
    tensor->mem = ptr;
    return tensor;
}

void device_cuda::tensor_copy(void *dest, void *src, int bytes, tensor_copy_kind copy_kind){
    if(copy_kind == TENSOR_COPY_D2H){
        tensor_t * t = (tensor_t *)src;
        CHECK_CUDA(cudaMemcpy(dest, t->mem, bytes, cudaMemcpyDeviceToHost));
    }
    else if(copy_kind == TENSOR_COPY_H2D){
        tensor_t * t = (tensor_t *)dest;
        CHECK_CUDA(cudaMemcpy(t->mem, src, bytes, cudaMemcpyHostToDevice));
    }
    else if (copy_kind == TENSOR_COPY_D2D){
        tensor_t * t1 = (tensor_t *)dest;
        tensor_t * t2 = (tensor_t *)src;
        CHECK_CUDA(cudaMemcpy(t1->mem, t2->mem, bytes, cudaMemcpyDeviceToDevice));
    }
}
void device_cuda::tensor_destroy(tensor_t * tensor){
    if(tensor->mem)
        CHECK_CUDA(cudaFree(tensor->mem));
    if(tensor->desc)
        CHECK_CUDNN(cudnnDestroyTensorDescriptor((cudnnTensorDescriptor_t)tensor->desc));
    delete tensor;
}
void device_cuda::tensor_set(tensor_t * tensor, unsigned char v){
    assert(tensor->mem);
    CHECK_CUDA(cudaMemset(tensor->mem, v, tensor->bytes()));
}
pooling_desc_t * device_cuda::pooling_desc_create(int * kernel, int * stride, int * padding, int n_dims,
    pooling_mode mode){
    assert(n_dims==2 && "miopen only support 2d pooling");

    cudnnPoolingMode_t pool_mode;
    cudnnPoolingDescriptor_t desc;

    pool_mode = to_cudnn_pooling_mode(mode);
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&desc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(desc, pool_mode, CUDNN_PROPAGATE_NAN,
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

void device_cuda::pooling_desc_destroy(pooling_desc_t * pooling_desc){
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor((cudnnPoolingDescriptor_t)pooling_desc->desc));
    delete pooling_desc;
}
activation_desc_t * device_cuda::activation_desc_create(activation_mode mode, float alpha){
    cudnnActivationDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&desc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(desc,
            to_cudnn_activation_mode(mode), CUDNN_NOT_PROPAGATE_NAN, (double)alpha));

    activation_desc_t * act_desc = new activation_desc_t;
    act_desc->mode = mode;
    act_desc->alpha = alpha;
    act_desc->desc = desc;
    return  act_desc;
}
activation_desc_t device_cuda::activation_desc_destroy(activation_desc_t * act_desc){
    CHECK_CUDNN(cudnnDestroyActivationDescriptor((cudnnActivationDescriptor_t)act_desc->desc));
    delete act_desc;
}