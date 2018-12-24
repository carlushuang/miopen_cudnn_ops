#include "operator.hpp"

op_pooling_cudnn::op_pooling_cudnn(void * desc) : op_pooling(desc) {
    forward_prepared = 0;
    backward_prepared = 0;

    workspace_tensor = nullptr;
}
op_pooling_cudnn::~op_pooling_cudnn(){
    if(workspace_tensor)
        dev->tensor_destroy(workspace_tensor);
}

void op_pooling_cudnn::forward(tensor_t * input, tensor_t * output)
{
    device_cuda * dev_cuda = (device_cuda *)dev;
    if(!forward_prepared){
        forward_prepared = 1;
    }
    float alpha = 1.f;
    float beta = 0.f;
    CHECK_CUDNN(cudnnPoolingForward(dev_cuda->handle,
        (const cudnnPoolingDescriptor_t)pooling_desc->desc,
        &alpha,
        (cudnnTensorDescriptor_t)input->desc, input->mem,
        &beta, (cudnnTensorDescriptor_t)output->desc, output->mem));
}

void op_pooling_cudnn::backward(tensor_t * input, tensor_t * output)
{

}