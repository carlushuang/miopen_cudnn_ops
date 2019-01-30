#include "operator.hpp"
#include "math.hpp"

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
    int dilation_h = conv_desc->dilation[0];
    int dilation_w = conv_desc->dilation[1];
    int groups   = conv_desc->groups;
    int filters  = conv_desc->k;

    // TODO: only 1x1 conv with 0 padding, 1 dilation may not use im2col
    bool need_im2col = true;

#if 0
    std::cout<<"n:"<<batch<<", c:"<<channel<<", h:"<<input_h<<", w:"<<input_w<<", k:"<<filters<<", g:"<<groups<<", kh:"<<kernel_h<<
        ", kw:"<<kernel_w<<", sh:"<<stride_h<<", sw:"<<stride_w<<", ph:"<<pad_h<<", pw:"<<pad_w<<", dh:"<<dilation_h<<", dw:"<<dilation_w<<
        ", out_h:"<<out_h<<", out_w:"<<out_w<<std::endl;
#endif

    if(!forward_prepared){
        forward_prepared = 1;
        if(need_im2col){
            fwd_workspace_size = out_h*out_w*kernel_h*kernel_w*channel*data_type_unit(input->data_type) / groups;
            fwd_workspace_mem = dev->ws->get(fwd_workspace_size, input->data_type);
        }
    }

    fwd_workspace_mem = dev->ws->get(fwd_workspace_size, input->data_type);

    int blas_m, blas_n, blas_k;
    blas_m = filters/groups;
    blas_n = out_h*out_w;
    blas_k = kernel_h*kernel_w*channel/groups;
    for(int n=0;n<batch;n++){
        for(int g=0;g<groups;g++){
            float * im_ptr = (float*)input->mem + n*input_h*input_w*channel + g*input_h*input_w*channel/groups;
            float * col_ptr = need_im2col?(float*)fwd_workspace_mem:(float*)im_ptr;
            float * filter_ptr = (float*)filter->mem + g*(filters/groups)*(channel/groups)*kernel_h*kernel_w;
            float * out_ptr = (float*)output->mem + n*out_h*out_w*filters + g*out_h*out_w*filters/groups;
            if(need_im2col){
                math::im2col(im_ptr, channel/groups, input_h, input_w,
                    dilation_h, dilation_w, kernel_h, kernel_w,
                    stride_h, stride_w, pad_h, pad_w,
                    col_ptr);
            }

            math::cblas_sgemm(math::CblasRowMajor, math::CblasNoTrans, math::CblasNoTrans,
                blas_m, blas_n, blas_k,
                1.0f,
                filter_ptr, blas_k,
                col_ptr, blas_n,
                .0f,
                out_ptr, blas_n);
        }
    }
}

void op_convolution::backward(){
    
}