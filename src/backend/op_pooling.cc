#include "operator.hpp"
#include <float.h>
#include <iostream>

void op_pooling::forward()
{
    assert(input && output);
    // pool 2d
    int batch   = input->dim[0];
    int channel = input->dim[1];
    int input_h = input->dim[2];
    int input_w = input->dim[3];
    int out_h   = output->dim[2];
    int out_w   = output->dim[3];

    int stride_h = pooling_desc->stride[0];
    int stride_w = pooling_desc->stride[1];
    int kernel_h = pooling_desc->kernel[0];
    int kernel_w = pooling_desc->kernel[1];
    int pad_h    = pooling_desc->padding[0];
    int pad_w    = pooling_desc->padding[1];
#if 0
    std::cout<<"pooling, ";
    std::cout<<"input:"<<batch<<"x"<<channel<<"x"<<input_h<<"x"<<input_w<<", "
             <<"output:"<<batch<<"x"<<channel<<"x"<<out_h<<"x"<<out_w<<", "
             <<"kernel:"<<kernel_h<<"x"<<kernel_w<<", stride:"<<stride_h<<"x"<<stride_w
             <<", pad:"<<pad_h<<"x"<<pad_w<<", mode:"<<pooling_desc->mode <<std::endl;
#endif
    float * input_ptr = (float*)input->mem;
    float * output_ptr = (float*)output->mem;

    bool exclusive = pooling_desc->mode == POOLING_AVG_EXCLUSIVE ? true:false;

    float init_value = (pooling_desc->mode == POOLING_MAX ||
                pooling_desc->mode == POOLING_MAX_DETERMINISTIC)?
                -FLT_MAX:0;

    auto update_func = 
        (pooling_desc->mode == POOLING_MAX ||
                pooling_desc->mode == POOLING_MAX_DETERMINISTIC)?
        [](const float & x, float & u){u=u>x?u:x;} :
        [](const float & x, float & u){u+=x;};
    
    auto finilize_func = 
        (pooling_desc->mode == POOLING_MAX ||
                pooling_desc->mode == POOLING_MAX_DETERMINISTIC)?
        [](const float & window_size, float & u){} :
        [](const float & window_size, float & u){u/=window_size;};

    for(int n=0;n<batch;n++){
        for(int c=0;c<channel;c++){
            for(int h=0;h<out_h;h++){
                int h_start = h*stride_h-pad_h;
                int h_end = MIN(h_start + kernel_h, input_h);
                h_start = MAX(h_start, 0);
                for(int w=0;w<out_w;w++){
                    int w_start = w*stride_w-pad_w;
                    int w_end = MIN(w_start + kernel_w, input_w);
                    w_start = MAX(w_start, 0);

                    float val = init_value;
                    for(int y=h_start;y<h_end;y++){
                        for(int x=w_start;x<w_end;x++){
                            int in_idx = n*channel*input_h*input_w + c*input_h*input_w + y*input_w + x;
                            float cur_input = input_ptr[in_idx];
                            update_func(cur_input, val);
                        }
                    }
                    int pool_size = exclusive ? (h_end - h_start) * (w_end - w_start)
                                      : kernel_h * kernel_w;
                    finilize_func(pool_size, val);
                    int out_idx = n*channel*out_h*out_w + c*out_h*out_w +
                                h*out_w + w;
                    output_ptr[out_idx] = val;
                }
            }
        }
    }
}
void op_pooling::backward()
{
    assert(input && output && input_grad && output_grad);
    // pool 2d
    int batch   = input->dim[0];
    int channel = input->dim[1];
    int input_h = input->dim[2];
    int input_w = input->dim[3];
    int out_h   = output->dim[2];
    int out_w   = output->dim[3];

    int stride_h = pooling_desc->stride[0];
    int stride_w = pooling_desc->stride[1];
    int kernel_h = pooling_desc->kernel[0];
    int kernel_w = pooling_desc->kernel[1];
    int pad_h    = pooling_desc->padding[0];
    int pad_w    = pooling_desc->padding[1];

    float * input_ptr = (float*)input->mem;
    float * output_ptr = (float*)output->mem;
    float * input_grad_ptr = (float*)input_grad->mem;
    float * output_grad_ptr = (float*)output_grad->mem;

    bool exclusive = pooling_desc->mode == POOLING_AVG_EXCLUSIVE ? true:false;

    if(pooling_desc->mode == POOLING_MAX 
        || pooling_desc->mode == POOLING_MAX_DETERMINISTIC)
    {
    // maxpool start
    for(int n=0;n<batch;n++){
        for(int c=0;c<channel;c++){
            for(int h=0;h<out_h;h++){
                int h_start = h*stride_h-pad_h;
                int h_end = MIN(h_start + kernel_h, input_h);
                h_start = MAX(h_start, 0);
                for(int w=0;w<out_w;w++){
                    int w_start = w*stride_w-pad_w;
                    int w_end = MIN(w_start + kernel_w, input_w);
                    w_start = MAX(w_start, 0);

                    bool found = false;
                    int out_idx = n*channel*out_h*out_w + c*out_h*out_w + h*out_w + w;
                    float out_target = output_ptr[out_idx];
                    for(int y=h_start;y<h_end && !found ;y++){
                        for(int x=w_start;x<w_end && !found ;x++){
                            int in_idx = n*channel*input_h*input_w + c*input_h*input_w + y*input_w + x;

                            float cur_input = input_ptr[in_idx];
                            // TODO: store idx in forward path to simplify found maxpool grad
                            if(cur_input == out_target ){
                                input_grad_ptr[in_idx] += output_grad_ptr[out_idx];
                                found = true;
                            }
                        }
                    }
                }
            }
        }
    }
    // maxpool end
    }else{
    // avgpool start
    for(int n=0;n<batch;n++){
        for(int c=0;c<channel;c++){
            for(int h=0;h<out_h;h++){
                int h_start = h*stride_h-pad_h;
                int h_end = MIN(h_start + kernel_h, input_h);
                h_start = MAX(h_start, 0);
                for(int w=0;w<out_w;w++){
                    int w_start = w*stride_w-pad_w;
                    int w_end = MIN(w_start + kernel_w, input_w);
                    w_start = MAX(w_start, 0);

                    int out_idx = n*channel*out_h*out_w + c*out_h*out_w + h*out_w + w;
                    int pool_size = exclusive ? (h_end - h_start) * (w_end - w_start)
                                      : kernel_h * kernel_w;
                    for(int y=h_start;y<h_end ;y++){
                        for(int x=w_start;x<w_end ;x++){
                            int in_idx = n*channel*input_h*input_w + c*input_h*input_w + y*input_w + x;
                            input_grad_ptr[in_idx] += output_grad_ptr[out_idx] / pool_size;
                        }
                    }
                }
            }
        }
    }
    // avgpool end
    }
}