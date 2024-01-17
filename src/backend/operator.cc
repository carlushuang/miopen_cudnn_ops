#include "operator.hpp"

#ifdef WITH_MIOPEN
#define DISPATCH_OP(op_type, op_name)           \
    case op_type:                               \
        if(device->type == DEVICE_HIP)          \
            op = new op_name ## _miopen(desc);  \
        else                                    \
            op = new op_name(desc);             \
    break;
#define IGNORE_OP(op_type, op_name)             \
    case op_type: break;
#endif

#ifdef WITH_CUDNN
#define DISPATCH_OP(op_type, op_name)           \
    case op_type:                               \
        if(device->type == DEVICE_CUDA)         \
            op = new op_name ## _cudnn(desc);   \
        else                                    \
            op = new op_name(desc);             \
    break;
#define IGNORE_OP(op_type, op_name)             \
    case op_type: break;
#endif

operator_base * operator_create(device_base *device, operator_type op_type, void * desc){
    operator_base * op = nullptr;
    switch(op_type){
        DISPATCH_OP(OP_CONV, op_convolution)
        IGNORE_OP(OP_POOLING, op_pooling)
        IGNORE_OP(OP_ACTIVATION, op_activation)
    }

    if(op){
        op->type = op_type;
        op->dev = device;
    }

    return op;
}
void operator_destroy(operator_base * op){
    delete op;
}
