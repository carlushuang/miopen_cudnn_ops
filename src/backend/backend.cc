#include "backend.hpp"


device_c::device_c(){
    this->type = DEVICE_C;
}
device_c::~device_c() {}
tensor_t * device_c::tensor_create(int * dims, int n_dim, 
                    tensor_data_type data_type, tensor_layout layout)
{

}
void device_c::tensor_copy(void *src, void *dest, int bytes, tensor_copy_kind copy_kind)
{

}
void device_c::tensor_destroy(tensor_t * tensor)
{

}

device_base  * device_create(device_type type, int dev_id){
    device_base * handle;
#ifdef WITH_MIOPEN
    if(type == DEVICE_HIP){
        handle = new device_hip(dev_id);
        return handle;
    }
#endif

#ifdef WITH_CUDNN
    if(type == DEVICE_CUDA){
        handle = new device_cuda(dev_id);
        return handle;
    }
#endif

    handle = new device_c;

    return handle;
}
void device_destroy(device_base* handle){
    delete handle;
}