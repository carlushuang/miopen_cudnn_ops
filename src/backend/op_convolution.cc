#include "operator.hpp"

static void im2col(const float* data_im, int num_outs, int im_height,
                        int im_width, int dilation_h, int dilation_w,
                        int filter_height, int filter_width, int stride_height,
                        int stride_width, int padding_height, int padding_width,
                        int col_height, int col_width, float* data_col)
{
    
}

void op_convolution::forward(){
    assert(input && output && filter);
    int batch   = input->dim[0];
    int channel = input->dim[1];
    int input_h = input->dim[2];
    int input_w = input->dim[3];
    int out_h   = output->dim[2];
    int out_w   = output->dim[3];

    int stride_h = conv_desc->stride[0];
    int stride_w = conv_desc->stride[1];
    int kernel_h = conv_desc->kernel[0];
    int kernel_w = conv_desc->kernel[1];
    int pad_h    = conv_desc->padding[0];
    int pad_w    = conv_desc->padding[1];
    int groups   = conv_desc->groups;


    if(!forward_prepared){
        forward_prepared = 1;
        fwd_workspace_size = out_h*out_w*kernel_h*kernel_w*channel*data_type_unit(input->data_type) / groups;
        fwd_workspace_mem = dev->ws->get(fwd_workspace_size, input->data_type);
    }

    for(int n=0;n<batch;n++){
        for(int g=0;g<groups;g++){

        }
    }

}
void op_convolution::backward(){
    
}