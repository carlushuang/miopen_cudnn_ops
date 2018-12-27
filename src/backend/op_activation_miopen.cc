#include "operator.hpp"

op_activation_miopen::op_activation_miopen(void * desc) : op_activation(desc) {

}
op_activation_miopen::~op_activation_miopen(){

}
void op_activation_miopen::forward(){
    assert(input && output);
    device_hip * dev_hip = (device_hip *)dev;

    float alpha = 1.0f;
    float beta = .0f;

    CHECK_MIO(miopenActivationForward((miopenHandle_t)dev_hip->handle,
        (const miopenActivationDescriptor_t)act_desc->desc,
        &alpha,
        (const miopenTensorDescriptor_t)input->desc, input->mem,
        &beta,
        (const miopenTensorDescriptor_t)output->desc, output->mem
    ));
}
void op_activation_miopen::backward(){
    assert(input && output && input_grad && output_grad);
    device_hip * dev_hip = (device_hip *)dev;

    float alpha = 1.0f;
    float beta = .0f;

    CHECK_MIO(miopenActivationBackward((miopenHandle_t)dev_hip->handle,
        (miopenActivationDescriptor_t)act_desc->desc,
        &alpha,
        (const miopenTensorDescriptor_t)output->desc, output->mem,
        (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
        (const miopenTensorDescriptor_t)input->desc, input->mem,
        &beta,
        (const miopenTensorDescriptor_t)input_grad->desc, input_grad->mem
    ));
}