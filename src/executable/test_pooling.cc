#include "backend.hpp"
#include "operator.hpp"

int main(){
    device_base * dev = device_create(DEVICE_HIP, 0);

    int pooling_kernel[2] = {2,2};
    int pooling_stride[2] = {2,2};
    int pooling_padding[2] = {0,0};
    pooling_desc_t * pooling_desc = dev->pooling_desc_create(
        pooling_kernel, pooling_stride, pooling_padding, 2,
        POOLING_MAX);
    operator_base * op_pooling = operator_create(dev, OP_POOLING, pooling_desc);

    int t_in_dim[4] = {2,3,128,128};
    tensor_t *t_in = dev->tensor_create(t_in_dim, 4,
            TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
    int t_out_dim[4];
    op_pooling->infer_shape(t_in, t_out_dim);
    tensor_t *t_out = dev->tensor_create(t_out_dim, 4,
            TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);

    op_pooling->forward(t_in, t_out);



    operator_destroy(op_pooling);
    dev->pooling_desc_destroy(pooling_desc);
    dev->tensor_destroy(t_in);
    dev->tensor_destroy(t_out);

    device_destroy(dev);
}