#ifndef __OP_POOLING_HPP
#define __OP_POOLING_HPP

class op_pooling:public operator_base{
public:
    op_pooling(void * desc){pooling_desc = (pooling_desc_t *)desc;}
    ~op_pooling(){}
    virtual void forward(tensor_t * input, tensor_t * output);
    virtual void backward(tensor_t * input, tensor_t * output);
    virtual void infer_shape(tensor_t * input, int * out_dim){
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
    virtual void forward(tensor_t * input, tensor_t * output);
    virtual void backward(tensor_t * input, tensor_t * output);

    tensor_t * workspace_tensor;

    int forward_prepared;
    int backward_prepared;
};
#endif

#endif