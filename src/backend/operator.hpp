#ifndef __OPERATOR_HPP
#define __OPERATOR_HPP

#include "backend.hpp"

enum operator_type{
    OP_CONV = 0,
    OP_POOLING,
    OP_ACTIVATION
};

class operator_base{
public:
    operator_base(){}
    virtual ~operator_base(){}
    virtual void forward() = 0;
    virtual void backward() = 0;

    virtual void infer_shape(int * out_dim){}

    tensor_t * input ={nullptr};
    tensor_t * output ={nullptr};
    tensor_t * filter ={nullptr};

    tensor_t * input_grad ={nullptr};
    tensor_t * output_grad ={nullptr};
    tensor_t * filter_grad ={nullptr};

    operator_type type;
    device_base * dev ={nullptr};

    int forward_prepared ={0};
    int backward_prepared ={0};
};
/*****************************************************************************
 * convolution op
*/

class op_convolution : public operator_base{
public:
    op_convolution(void * desc){conv_desc = (convolution_desc_t*)desc;}
    ~op_convolution(){}
    virtual void forward();
    virtual void backward();
    virtual void infer_shape(int * out_dim){
        assert(input && conv_desc);
        out_dim[0] = input->dim[0];
        out_dim[1] = conv_desc->k;
        // https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
        auto conv_size_func = [](int in_size, int pad, int dilation, int ksize, int stride){
            return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
        };
        int dilation[2] = {1, 1};
        out_dim[2] = conv_size_func(input->dim[2], conv_desc->padding[0],
                dilation[0], conv_desc->kernel[0], conv_desc->stride[0]);
        out_dim[3] = conv_size_func(input->dim[3], conv_desc->padding[1],
                dilation[1], conv_desc->kernel[1], conv_desc->stride[1]);
    }

    convolution_desc_t * conv_desc;

    size_t fwd_workspace_size;
    void * fwd_workspace_mem;

    size_t bwd_data_workspace_size;
    void * bwd_data_workspace_mem;
    size_t bwd_filter_workspace_size;
    void * bwd_filter_workspace_mem;
};
#ifdef WITH_MIOPEN
class op_convolution_miopen : public op_convolution{
public:
   op_convolution_miopen(void * desc);
    ~op_convolution_miopen();
    virtual void forward();
    virtual void backward();
    miopenConvFwdAlgorithm_t fwd_algo;
    miopenConvBwdWeightsAlgorithm_t bwd_weights_algo;
    miopenConvBwdDataAlgorithm_t bwd_data_algo;
};
#endif
#ifdef WITH_CUDNN
class op_convolution_cudnn : public op_convolution{
public:
   op_convolution_cudnn(void * desc);
    ~op_convolution_cudnn();
    virtual void forward();
    virtual void backward();
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionFwdAlgo_t fwd_algo;
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;

};
#endif

/*****************************************************************************
 * pooling op
*/
class op_pooling:public operator_base{
public:
    op_pooling(void * desc){pooling_desc = (pooling_desc_t *)desc;}
    virtual ~op_pooling(){}
    virtual void forward();
    virtual void backward();
    virtual void infer_shape(int * out_dim){
        assert(input);
        out_dim[0] = input->dim[0];
        out_dim[1] = input->dim[1];
        int ksize,pad,stride,in_size;
        ksize      = pooling_desc->kernel[0];
        pad        = pooling_desc->padding[0];
        stride     = pooling_desc->stride[0];
        in_size    = input->dim[2];
        out_dim[2] = (in_size - ksize + 2 * pad) / stride + 1;

        ksize      = pooling_desc->kernel[1];
        pad        = pooling_desc->padding[1];
        stride     = pooling_desc->stride[1];
        in_size    = input->dim[3];
        out_dim[3] = (in_size - ksize + 2 * pad) / stride + 1;
    }

    pooling_desc_t * pooling_desc;
};

#ifdef WITH_MIOPEN
class op_pooling_miopen : public op_pooling{
public:
    op_pooling_miopen(void * desc);
    ~op_pooling_miopen();
    virtual void forward();
    virtual void backward();

    size_t workspace_size;
    void * workspace_mem;
};
#endif
#ifdef WITH_CUDNN
class op_pooling_cudnn : public op_pooling{
public:
    op_pooling_cudnn(void * desc);
    ~op_pooling_cudnn();
    virtual void forward();
    virtual void backward();

};
#endif

/*****************************************************************************
 * activation op
*/
class op_activation:public operator_base{
public:
    op_activation(void * desc){act_desc = (activation_desc_t *)desc;}
    virtual ~op_activation(){}
    virtual void forward();
    virtual void backward();
    virtual void infer_shape(int * out_dim){
        assert(input);
        out_dim[0] = input->dim[0];
        out_dim[1] = input->dim[1];
        out_dim[2] = input->dim[2];
        out_dim[3] = input->dim[3];
    }

    activation_desc_t * act_desc;
};

#ifdef WITH_MIOPEN
class op_activation_miopen : public op_activation{
public:
    op_activation_miopen(void * desc);
    ~op_activation_miopen();
    virtual void forward();
    virtual void backward();

};
#endif
#ifdef WITH_CUDNN
class op_activation_cudnn : public op_activation{
public:
    op_activation_cudnn(void * desc);
    ~op_activation_cudnn();
    virtual void forward();
    virtual void backward();

};
#endif

/*****************************************************************************
 */
operator_base * operator_create(device_base *device, operator_type op_type, void * desc);
void operator_destroy(operator_base * op);

#endif
