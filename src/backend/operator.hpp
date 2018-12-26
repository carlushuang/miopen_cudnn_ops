#ifndef __OPERATOR_HPP
#define __OPERATOR_HPP

#include "backend.hpp"

enum operator_type{
    OP_CONV,
    OP_POOLING,
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

    tensor_t * input_grad ={nullptr};
    tensor_t * output_grad ={nullptr};
    tensor_t * filter_grad ={nullptr};

    operator_type type;
    device_base * dev ={nullptr};

    tensor_t * workspace_tensor ={nullptr};

    int forward_prepared ={0};
    int backward_prepared ={0};
};

#include "op_pooling.hpp"

operator_base * operator_create(device_base *device, operator_type op_type, void * desc);
void operator_destroy(operator_base * op);

#endif