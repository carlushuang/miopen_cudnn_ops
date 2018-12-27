#include "operator.hpp"

op_activation_cudnn::op_activation_cudnn(void * desc) : op_activation(desc) {

}
op_activation_cudnn::~op_activation_cudnn(){

}
void op_activation_cudnn::forward(){
    assert(input && output);
    device_cuda * dev_cuda = (device_cuda *)dev;

    float alpha = 1.0f;
    float beta = .0f;
    if(act_desc->mode == ACTIVATION_IDENTITY){
        LOG_E()<<"CUDNN_ACTIVATION_IDENTITY not intented for use in act forward/backeard"<<std::endl;
        LOG_E()<<"It is for cudnnConvolutionBiasActivationForward"<<std::endl;
        return ;
    }
    CHECK_CUDNN(cudnnActivationForward((cudnnHandle_t)dev_cuda->handle,
        (cudnnActivationDescriptor_t)act_desc->desc,
        &alpha,
        (const cudnnTensorDescriptor_t)input->desc, input->mem,
        &beta,
        (const cudnnTensorDescriptor_t)output->desc, output->mem
    ));
}

void op_activation_cudnn::backward(){
    assert(input && output && input_grad && output_grad);
    device_cuda * dev_cuda = (device_cuda *)dev;

    float alpha = 1.0f;
    float beta = .0f;
    if(act_desc->mode == ACTIVATION_IDENTITY){
        LOG_E()<<"CUDNN_ACTIVATION_IDENTITY not intented for use in act forward/backeard"<<std::endl;
        LOG_E()<<"It is for cudnnConvolutionBiasActivationForward"<<std::endl;
        return ;
    }
    CHECK_CUDNN(cudnnActivationBackward((cudnnHandle_t)dev_cuda->handle,
        (cudnnActivationDescriptor_t)act_desc->desc,
        &alpha,
        (const cudnnTensorDescriptor_t)output->desc, output->mem,
        (const cudnnTensorDescriptor_t)output_grad->desc, output_grad->mem,
        (const cudnnTensorDescriptor_t)input->desc, input->mem,
        &beta,
        (const cudnnTensorDescriptor_t)input_grad->desc, input_grad->mem
    ));
}