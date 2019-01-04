#include "backend.hpp"
#include <stdlib.h>
#include <string.h>

std::ostream & log_to_stream(log_level level){
    if(level == LOG_INFO){
        std::cout<<"[INFO] ";
        return std::cout;
    }
    if(level == LOG_WARNING){
        std::cerr<<"[WARN] ";
        return std::cerr;
    }
    if(level == LOG_ERROR){
        std::cerr<<"[ERRO] ";
        return std::cerr;
    }
    if(level == LOG_FATAL){
        std::cerr<<"[FATAL]";
        return std::cerr;
    }
    return std::cout;  
}

workspace::workspace(device_base * dev_):dev(dev_){}
workspace::~workspace(){
    // CAUTION, device must not destroyed
    if(workspace_tensor)
        dev->tensor_destroy(workspace_tensor);
}

void * workspace::get(int bytes, tensor_data_type dt){
    return get_tensor(bytes, dt)->mem;
}
tensor_t * workspace::get_tensor(int bytes, tensor_data_type dt){
    assert(bytes != 0);
    if(!workspace_tensor){
        workspace_tensor = dev->tensor_create(&bytes, 1, dt, TENSOR_LAYOUT_1D);
        cur_byte = bytes;
    }
    else{
        if(bytes > cur_byte){
            dev->tensor_destroy(workspace_tensor);
            workspace_tensor = dev->tensor_create(&bytes, 1, dt, TENSOR_LAYOUT_1D);
            cur_byte = bytes;
        }
    }
    // CAUTION: the inner size of this tensor may bigger than request!
    return workspace_tensor;
}

device_c::device_c(){
    this->type = DEVICE_C;
}
device_c::~device_c() {}
tensor_t * device_c::tensor_create(int * dims, int n_dim, 
                    tensor_data_type data_type, tensor_layout layout)
{
    int elem = 1;
    for(int i=0;i<n_dim;i++)
        elem *= dims[i];
    int total_byte = elem * data_type_unit(data_type);
    void * ptr = (void*)new unsigned char[total_byte];
    assert(ptr);

    tensor_t * tensor = new tensor_t;
    //std::cout<<"create ptr:"<<ptr<<", bytes:"<<total_byte<<", obj:"<<(void*)tensor<<std::endl;
    tensor->dim[0] = dims[0];
    tensor->dim[1] = n_dim>1?dims[1]:0;
    tensor->dim[2] = n_dim>2?dims[2]:0;
    tensor->dim[3] = n_dim>3?dims[3]:0;
    tensor->n_dims = n_dim;
    tensor->data_type = data_type;
    tensor->layout = layout;
    tensor->desc = nullptr;
    tensor->mem = ptr;
    return tensor;
}
void device_c::tensor_copy(void *dest, void *src, int bytes, tensor_copy_kind copy_kind)
{
    (void)copy_kind;
    tensor_t * t_dest = (tensor_t*)dest;
    tensor_t * t_src = (tensor_t*)src;
    memcpy(t_dest->mem, t_src->mem, bytes);
}
void device_c::tensor_destroy(tensor_t * tensor)
{
    if(tensor->mem)
        delete [] (unsigned char*)tensor->mem;
    delete tensor;
}
void device_c::tensor_set(tensor_t * tensor, unsigned char v){
    assert(tensor->mem);
    memset(tensor->mem, v, tensor->bytes());
}
device_base  * device_create(device_type type, int dev_id){
    device_base * handle=nullptr;
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
    if(type == DEVICE_C)
        handle = new device_c;

    return handle;
}
void device_destroy(device_base* handle){
    handle->shutdown();
    delete handle;
}