#include "operator.hpp"
#include <float.h>

void op_pooling::forward(tensor_t * input, tensor_t * output)
{
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
                            int in_idx = n*channel*input_h*input_w +
                                    c*input_h*input_w + y*input_w + x;
                            update_func(input_ptr[in_idx], val);
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
void op_pooling::backward(tensor_t * input, tensor_t * output)
{

}