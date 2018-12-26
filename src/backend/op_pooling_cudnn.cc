#include "operator.hpp"

op_pooling_cudnn::op_pooling_cudnn(void * desc) : op_pooling(desc) {

}
op_pooling_cudnn::~op_pooling_cudnn(){
    if(workspace_tensor)
        dev->tensor_destroy(workspace_tensor);
}

void op_pooling_cudnn::forward()
{
    assert(input && output);
    device_cuda * dev_cuda = (device_cuda *)dev;

    float alpha = 1.f;
    float beta = 0.f;
    CHECK_CUDNN(cudnnPoolingForward(dev_cuda->handle,
        (const cudnnPoolingDescriptor_t)pooling_desc->desc,
        &alpha,
        (cudnnTensorDescriptor_t)input->desc, input->mem,
        &beta, (cudnnTensorDescriptor_t)output->desc, output->mem));
}

void op_pooling_cudnn::backward()
{
    assert(input && output && input_grad && output_grad);
    float alpha = 1.f;
    float beta = 0.f;
    device_cuda * dev_cuda = (device_cuda *)dev;
    CHECK_CUDNN(cudnnPoolingBackward(dev_cuda->handle,
        (const cudnnPoolingDescriptor_t)pooling_desc->desc,
        &alpha,
        (const cudnnTensorDescriptor_t)output->desc, output->mem,
        (const cudnnTensorDescriptor_t)output_grad->desc, output_grad->mem,
        (const cudnnTensorDescriptor_t)input->desc, input->mem,
        &beta,
        (const cudnnTensorDescriptor_t)input_grad->desc, input_grad->mem));
}