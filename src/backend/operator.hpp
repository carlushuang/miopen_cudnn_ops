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
    virtual void forward(tensor_t * input, tensor_t * output) = 0;
    virtual void backward(tensor_t * input, tensor_t * output) = 0;

    virtual void infer_shape(tensor_t * input, int * out_dim){}

    operator_type type;
    device_base *  dev;
};

#include "op_pooling.hpp"

operator_base * operator_create(device_base *device, operator_type op_type, void * desc);
void operator_destroy(operator_base * op);

#endif