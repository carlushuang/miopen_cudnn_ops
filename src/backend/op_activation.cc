#include "operator.hpp"
#include <math.h>
#include <algorithm>

// https://en.wikipedia.org/wiki/Activation_function
static float act_sigmoid(float x) {return 1 / (1 + expf(-x));}
static float act_sigmoid_grad(float x, float y) { (void)x;return y*(1-y);}

static float act_relu(float x){return std::max(.0f, x);}
static float act_relu_grad(float x,float y)
{
    (void)y;
    // x>=0?1:0, not valid in cudnn
    return x>0?1:0; // valid in cudnn
}

static float act_tanh(float x){return (expf(x)-expf(-x))/(expf(x)+expf(-x)); }
static float act_tanh_grad(float x,float y){return 1-y*y;}

static float act_clipped_relu(float x, float clip){return std::min(std::max(.0f, x), clip);}
static float act_clipped_relu_grad(float x, float y, float clip)
            {(void)y; return (x>clip||x<=0)?0:1;  /*valid in cudnn*/ }

static float act_elu(float x, float alpha){return x>0?x:alpha*(expf(x)-1);}
static float act_elu_grad(float x,float y, float alpha){return x>=0?1:(alpha+y);}

static float act_identity(float x){return x;}
static float act_identity_grad(float x, float y){(void)x;(void)y;return 1;}

void op_activation::forward(){
    assert(input && output);
    assert(input->elem() == output->elem());
    int elem = input->elem();
    float alpha = act_desc->alpha;
    float * input_ptr = (float*)input->mem;
    float * output_ptr = (float*)output->mem;
    switch(act_desc->mode){
        case ACTIVATION_SIGMOID:{
            for(int i=0;i<elem;i++){
                output_ptr[i] = act_sigmoid(input_ptr[i]);
            }
        }
        break;
        case ACTIVATION_RELU:{
            for(int i=0;i<elem;i++){
                output_ptr[i] = act_relu(input_ptr[i]);
            }
        }
        break;
        case ACTIVATION_TANH:{
            for(int i=0;i<elem;i++){
                output_ptr[i] = act_tanh(input_ptr[i]);
            }
        }
        break;
        case ACTIVATION_CLIPPED_RELU:{
            for(int i=0;i<elem;i++){
                output_ptr[i] = act_clipped_relu(input_ptr[i], alpha);
            }
        }
        break;
        case ACTIVATION_ELU:{
            for(int i=0;i<elem;i++){
                output_ptr[i] = act_elu(input_ptr[i], alpha);
            }
        }
        break;
        case ACTIVATION_IDENTITY:{
            for(int i=0;i<elem;i++){
                output_ptr[i] = act_identity(input_ptr[i]);
            }
        }
        break;
        default:
            assert(0 && "no such pooling mode in forward");
        break;
    }
}

void op_activation::backward(){
    assert(input && output && input_grad && output_grad);
    assert(input->elem() == output->elem());
    int elem = input->elem();
    float alpha = act_desc->alpha;
    float * input_ptr = (float*)input->mem;
    float * output_ptr = (float*)output->mem;
    float * input_grad_ptr = (float*)input_grad->mem;
    float * output_grad_ptr = (float*)output_grad->mem;
    switch(act_desc->mode){
        case ACTIVATION_SIGMOID:{
            for(int i=0;i<elem;i++){
                input_grad_ptr[i] = output_grad_ptr[i] *
                        act_sigmoid_grad(input_ptr[i], output_ptr[i]);
            }
        }
        break;
        case ACTIVATION_RELU:{
            for(int i=0;i<elem;i++){
                input_grad_ptr[i] = output_grad_ptr[i] *
                        act_relu_grad(input_ptr[i], output_ptr[i]);
            }
        }
        break;
        case ACTIVATION_TANH:{
            for(int i=0;i<elem;i++){
                input_grad_ptr[i] = output_grad_ptr[i] *
                        act_tanh_grad(input_ptr[i], output_ptr[i]);
            }
        }
        break;
        case ACTIVATION_CLIPPED_RELU:{
            for(int i=0;i<elem;i++){
                input_grad_ptr[i] = output_grad_ptr[i] *
                        act_clipped_relu_grad(input_ptr[i], output_ptr[i], alpha);
            }
        }
        break;
        case ACTIVATION_ELU:{
            for(int i=0;i<elem;i++){
                input_grad_ptr[i] = output_grad_ptr[i] *
                        act_elu_grad(input_ptr[i], output_ptr[i], alpha);
            }
        }
        break;
        case ACTIVATION_IDENTITY:{
            for(int i=0;i<elem;i++){
                input_grad_ptr[i] = output_grad_ptr[i] *
                        act_identity_grad(input_ptr[i], output_ptr[i]);
            }
        }
        break;
        default:
            assert(0 && "no such pooling mode in backward");
        break;
    }
}
