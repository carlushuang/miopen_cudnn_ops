#include "backend.hpp"
#include "operator.hpp"
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <assert.h>
#include <unordered_map>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>

static device_base * determin_device(){
    std::string backend = "HIP";
    int device_id = 0;
    char * var = getenv("BACKEND");
    if(var && !strcmp(var, "HIP"))
        backend = "HIP";
    if(var && !strcmp(var, "CUDA"))
        backend = "HIP";

    var = getenv("DEVICE_ID");
    if(var)
        device_id = atoi(var);

    device_base * dev = nullptr;
    if(backend == "HIP")
        dev = device_create(DEVICE_HIP, device_id);
    else if(backend == "CUDA")
        dev = device_create(DEVICE_CUDA, device_id);

    assert(dev && "Fail to create device");
    return dev;
}
static void rand_float(float * vec, int len){
    int i;

    static int flag = 0;
    if(!flag){ srand(time(NULL)); flag = 1; }

    for(i=0;i<len;i++){
        vec[i] = ((float)(rand() % 1000)) / 1000.0f;
    }
}

// parse int arg value
class arg_parser{
#define ARG_VALUE_INIT 32767
public:
    struct arg_store{
        std::string arg_name;
        int value;
        int default_value;
        std::string help_str;
    };
    arg_parser(const char * _name):name(_name){};
    ~arg_parser(){}
    void insert_arg(const char * arg, const char * help, int default_value){
        arg_store a;
        a.arg_name = std::string("-") + arg;
        a.help_str = help;
        a.default_value = default_value;
        a.value = ARG_VALUE_INIT;
        arg_pair[a.arg_name] = a;
    }
    bool parse(int argc, char ** argv){
        for(int i=0;i<argc;i+=2){
            std::string arg_name = argv[i];
            if(arg_pair.count(arg_name) == 0){
                std::cerr<<"unrecognized arg "<<arg_name<<std::endl;;
                usage();
                return false;
            }
            if((i+1) >= argc){
                std::cerr<<"no value specified for this arg"<<std::endl;
                usage();
                return false;
            }
            int v = atoi(argv[i+1]);
            arg_pair[arg_name].value = v;
        }
        return true;
    }
    int  get_arg(const char * arg){
        std::string arg_name = std::string("-") + arg;
        if(arg_pair.count(arg_name) == 0){
            std::cerr<<"no such arg "<<arg_name<<std::endl;;
            usage();
            return false;
        }
        int v = arg_pair[arg_name].value;
        if(v == ARG_VALUE_INIT)
            v = arg_pair[arg_name].default_value;
        return v;
    }
    void usage(){
        std::cout<<name<<" args:"<<std::endl;
        for(auto & it : arg_pair){
            arg_store a = it.second;
            std::cout<<"    "<<it.first<<", "<<a.help_str<<
                "(default:"<<a.default_value<<")"
#if 1
                <<", cur:"<<a.value
#endif           
                <<std::endl;
        }
    }
private:
    std::string name;
    std::unordered_map<std::string, arg_store> arg_pair;
};

static device_base * gpu_dev;
static device_base * cpu_dev;

static int pooling_driver(int argc, char ** argv){
    arg_parser parser("pooling");
    parser.insert_arg("k", "kernel size", 2);
    parser.insert_arg("s", "stride", 2);
    parser.insert_arg("p", "padding", 0);
    parser.insert_arg("n", "batch", 2);
    parser.insert_arg("c", "channel", 3);
    parser.insert_arg("h", "height", 128);
    parser.insert_arg("w", "width", 128);
    parser.insert_arg("m", "pooling mode, "
                "0-MAX 1-MAX_DETERMINISTIC "
                "2-AVG_EXCLUSIVE 3-AVG_INCLUSIVE ", 0);
    parser.parse(argc, argv);

    parser.usage();

    int ksize = parser.get_arg("k");
    int psize = parser.get_arg("p");
    int ssize = parser.get_arg("s");
    int n     = parser.get_arg("n");
    int c     = parser.get_arg("c");
    int h     = parser.get_arg("h");
    int w     = parser.get_arg("w");
    int pmode = parser.get_arg("m");

    int pooling_kernel[2] = {ksize,ksize};
    int pooling_stride[2] = {ssize,ssize};
    int pooling_padding[2] = {psize,psize};
    pooling_mode pm = POOLING_MAX;
    if(pmode == 0) pm = POOLING_MAX;
    else if(pmode == 1) pm = POOLING_MAX_DETERMINISTIC;
    else if(pmode == 2) pm = POOLING_AVG_EXCLUSIVE;
    else if(pmode == 3) pm = POOLING_AVG_INCLUSIVE;

    pooling_desc_t * pooling_desc = gpu_dev->pooling_desc_create(
        pooling_kernel, pooling_stride, pooling_padding, 2,
        pm);
    operator_base * op_pooling = operator_create(gpu_dev, OP_POOLING, pooling_desc);

    int t_in_dim[4] = {n,c,h,w};
    tensor_t *t_in = gpu_dev->tensor_create(t_in_dim, 4,
            TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
    int t_out_dim[4];
    op_pooling->infer_shape(t_in, t_out_dim);
    tensor_t *t_out = gpu_dev->tensor_create(t_out_dim, 4,
            TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);

    // create cpu side op
    operator_base * op_pooling_c = operator_create(cpu_dev, OP_POOLING, pooling_desc);
    tensor_t * t_in_c = cpu_dev->tensor_create(t_in_dim, 4,
            TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
    tensor_t * t_out_c = cpu_dev->tensor_create(t_out_dim, 4,
            TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
    rand_float((float*)t_in_c->mem, t_in_c->elem());
    gpu_dev->tensor_copy(t_in, t_in_c->mem, t_in_c->bytes(), TENSOR_COPY_H2D);

    op_pooling->forward(t_in, t_out);

    //validation
    op_pooling_c->forward(t_in_c, t_out_c);

    // compare
    float * dev_out = new float[t_out->elem()];
    gpu_dev->tensor_copy(dev_out, t_out, t_out->bytes(), TENSOR_COPY_D2H);

    int error_cnt = util_compare_data(dev_out, t_out_c->mem, t_out_c->elem(), TENSOR_DT_FLOAT, 0.01);
    if(error_cnt){
        std::cout<<"compare fail"<<std::endl;
    }else{
        std::cout<<"result verified"<<std::endl;
    }

    operator_destroy(op_pooling);
    gpu_dev->pooling_desc_destroy(pooling_desc);
    gpu_dev->tensor_destroy(t_in);
    gpu_dev->tensor_destroy(t_out);

    operator_destroy(op_pooling_c);
    cpu_dev->tensor_destroy(t_in_c);
    cpu_dev->tensor_destroy(t_out_c);
    return 0;
}

int main(int argc, char ** argv){
    gpu_dev = determin_device();
    cpu_dev = device_create(DEVICE_C, 0);
    if(argc<=1){
        std::cout<<"arg_store not enought"<<std::endl;
        return -1;
    }
    std::string op_type = argv[1];
    argv += 2;
    argc -= 2;
    int rtn = 0;
    if(op_type == "pooling")
        rtn = pooling_driver(argc, argv);


    device_destroy(gpu_dev);
    device_destroy(cpu_dev);
    return rtn;
}