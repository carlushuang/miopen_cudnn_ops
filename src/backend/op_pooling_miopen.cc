#include "operator.hpp"

op_pooling_miopen::op_pooling_miopen(void * desc) : op_pooling(desc) {

}
op_pooling_miopen::~op_pooling_miopen(){

}


void op_pooling_miopen::forward()
{
    assert(input && output);
    device_hip * dev_hip = (device_hip *)dev;
    if(!forward_prepared){
        CHECK_MIO(miopenPoolingGetWorkSpaceSize((const miopenTensorDescriptor_t)output->desc,
                &workspace_size));

        workspace_mem = workspace_size?
                            dev->ws->get(workspace_size, input->data_type):
                            nullptr;
        forward_prepared = 1;
    }
    float alpha = 1.f;
    float beta = 0.f;
    workspace_mem = workspace_size?
                            dev->ws->get(workspace_size, input->data_type):
                            nullptr;
    CHECK_MIO(miopenPoolingForward(dev_hip->handle,
        (const miopenPoolingDescriptor_t)pooling_desc->desc,
        &alpha,
        (const miopenTensorDescriptor_t)input->desc, input->mem,
        &beta,
        (const miopenTensorDescriptor_t)output->desc, output->mem, true,
        workspace_mem, workspace_size));
}

void op_pooling_miopen::backward()
{
    assert(input && output && input_grad && output_grad);
    device_hip * dev_hip = (device_hip *)dev;
    float alpha = 1.f;
    float beta = 0.f;
    CHECK_MIO(miopenPoolingBackward(dev_hip->handle,
        (const miopenPoolingDescriptor_t)pooling_desc->desc,
        &alpha,
        (const miopenTensorDescriptor_t)output->desc, output->mem,
        (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
        (const miopenTensorDescriptor_t)input->desc, input->mem,
        &beta,
        (const miopenTensorDescriptor_t)input_grad->desc, input_grad->mem,
        workspace_mem));
}
