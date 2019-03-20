#include "operator.hpp"

op_convolution_cudnn::op_convolution_cudnn(void * desc) : op_convolution(desc){

}
op_convolution_cudnn::~op_convolution_cudnn(){
    if(forward_prepared){
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    }
}

#define ALGO_CASE_STR(algo) \
    case algo: \
    return  #algo; \
    break

static const char * to_cudnn_fwd_algo_name(cudnnConvolutionFwdAlgo_t fwd_algo){
    switch(fwd_algo){
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_GEMM);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
        default:
            return "N/A";
        break;
    }
}

static const char * to_cudnn_bwd_data_algo_name(cudnnConvolutionBwdDataAlgo_t bwd_data_algo){
    switch(bwd_data_algo){
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED);
        default:
            return "N/A";
        break;
    }
}

static const char * to_cudnn_bwd_filter_algo_name(cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo){
    switch(bwd_filter_algo){
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED);
        ALGO_CASE_STR(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING);
        default:
            return "N/A";
        break;
    }
}


void op_convolution_cudnn::forward(){
    assert(input && output && filter);
    device_cuda * dev_cuda = (device_cuda *)dev;

    if(!forward_prepared){
        forward_prepared = 1;
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc,
	                to_cudnn_data_type(filter->data_type),
                    /* CUDNN_TENSOR_NCHW->KCRS, K->out feature map, C->input feat map, R->row per filter, S->col per filter*/
                    to_cudnn_layout(filter->layout), 
                    conv_desc->k,
                    //https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#grouped-convolutions
                    conv_desc->input_c/conv_desc->groups, 
                    conv_desc->kernel[0], conv_desc->kernel[1]));

        // TODO: set CUDNN_TENSOR_OP_MATH can use tensor core if available, here disable it
        CHECK_CUDNN(cudnnSetConvolutionMathType((const cudnnConvolutionDescriptor_t)conv_desc->desc,
                CUDNN_DEFAULT_MATH));

        // find fwd algo
        cudnnConvolutionFwdAlgoPerf_t perfs[4];
        int returned_algos;

        CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(dev_cuda->handle,
            (const cudnnTensorDescriptor_t)input->desc,
            filter_desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)output->desc,
            4, &returned_algos, perfs));

#if 0
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(dev_cuda->handle,
            (const cudnnTensorDescriptor_t)input->desc,
            filter_desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)output->desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            0,
            &algo));
#endif
#if 1
        LOG_I()<<" found cudnnConv "<<returned_algos<<" fwd algo, using "<<perfs[0].algo<<"("<<
            to_cudnn_fwd_algo_name(perfs[0].algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs[i].algo<< "(" <<to_cudnn_fwd_algo_name(perfs[i].algo)
                 << ") - time: " << perfs[i].time << ", Memory: " << perfs[i].memory<<std::endl;
        }
#endif
        fwd_algo = perfs[0].algo;

        // find workspace
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(dev_cuda->handle,
            (const cudnnTensorDescriptor_t)input->desc,
            filter_desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)output->desc, fwd_algo, &fwd_workspace_size));
        fwd_workspace_mem = fwd_workspace_size?
                            dev->ws->get(fwd_workspace_size, input->data_type):
                            nullptr;
    }
#if 0
    dump_cudnn_convolution_desc((const cudnnConvolutionDescriptor_t)conv_desc->desc);
    dump_cudnn_filter_desc(filter_desc);
    dump_cudnn_tensor_desc((const cudnnTensorDescriptor_t)input->desc);
    dump_cudnn_tensor_desc((const cudnnTensorDescriptor_t)filter->desc);
    dump_cudnn_tensor_desc((const cudnnTensorDescriptor_t)output->desc);
#endif
    float alpha = 1.f;
    float beta = .0f;
    fwd_workspace_mem = fwd_workspace_size?
                        dev->ws->get(fwd_workspace_size, input->data_type):
                        nullptr;
    CHECK_CUDNN(cudnnConvolutionForward(dev_cuda->handle,
            &alpha,
            (const cudnnTensorDescriptor_t)input->desc, input->mem,
            filter_desc, filter->mem,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            fwd_algo, fwd_workspace_mem, fwd_workspace_size,
            &beta,
            (const cudnnTensorDescriptor_t)output->desc, output->mem));
}
void op_convolution_cudnn::backward_data(){
    assert(input && output && filter && input_grad && output_grad && filter_grad);
    device_cuda * dev_cuda = (device_cuda *)dev;
    if(!backward_data_prepared){
        backward_data_prepared = 1;

        // find bwd data algo
        cudnnConvolutionBwdDataAlgoPerf_t perfs_data[5];
        int returned_algos;
        CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(dev_cuda->handle, 
            filter_desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)input_grad->desc,
            5, &returned_algos, perfs_data));

        LOG_I()<<" found cudnnConv "<<returned_algos<<" bwd_data algo, using "<<perfs_data[0].algo<<"("<<
            to_cudnn_bwd_data_algo_name(perfs_data[0].algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs_data[i].algo<< "(" <<to_cudnn_bwd_data_algo_name(perfs_data[i].algo)
                 << ") - time: " << perfs_data[i].time << ", Memory: " << perfs_data[i].memory<<std::endl;
        }
        bwd_data_algo = perfs_data[0].algo;

        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(dev_cuda->handle,
            filter_desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)input_grad->desc,
            bwd_data_algo, &bwd_data_workspace_size));
        bwd_data_workspace_mem = bwd_data_workspace_size?
                                dev->ws->get(bwd_data_workspace_size, input->data_type):
                                nullptr;
    }
    float alpha = 1.f;
    float beta = 0.f;
    bwd_data_workspace_mem = bwd_data_workspace_size?
                                dev->ws->get(bwd_data_workspace_size, input->data_type):
                                nullptr;
    CHECK_CUDNN(cudnnConvolutionBackwardData(dev_cuda->handle,
            &alpha,
            (const cudnnFilterDescriptor_t)filter_desc, filter->mem,
            (const cudnnTensorDescriptor_t)output_grad->desc, output_grad->mem,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            bwd_data_algo, bwd_data_workspace_mem, bwd_data_workspace_size,
            &beta,
            (const cudnnTensorDescriptor_t)input_grad->desc, input_grad->mem));
}
void op_convolution_cudnn::backward_filter(){
    assert(input && output && filter && input_grad && output_grad && filter_grad);
    device_cuda * dev_cuda = (device_cuda *)dev;
    if(!backward_filter_prepared){
        backward_filter_prepared = 1;

        // find bwd filter algo
        cudnnConvolutionBwdFilterAlgoPerf_t perfs_filter[5];
        int returned_algos;
        CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(dev_cuda->handle, 
            (const cudnnTensorDescriptor_t)input->desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            filter_desc,
            5, &returned_algos, perfs_filter));

        LOG_I()<<" found cudnnConv "<<returned_algos<<" bwd_filter algo, using "<<perfs_filter[0].algo<<"("<<
            to_cudnn_bwd_filter_algo_name(perfs_filter[0].algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs_filter[i].algo<< "(" <<to_cudnn_bwd_filter_algo_name(perfs_filter[i].algo)
                 << ") - time: " << perfs_filter[i].time << ", Memory: " << perfs_filter[i].memory<<std::endl;
        }
        bwd_filter_algo = perfs_filter[0].algo;

        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(dev_cuda->handle,
            (const cudnnTensorDescriptor_t)input->desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            filter_desc,
            bwd_filter_algo, &bwd_filter_workspace_size));
        bwd_filter_workspace_mem = bwd_filter_workspace_size?
                                dev->ws->get(bwd_filter_workspace_size, input->data_type):
                                nullptr;
    }
    float alpha = 1.f;
    float beta = 0.f;
    bwd_filter_workspace_mem = bwd_filter_workspace_size?
                                dev->ws->get(bwd_filter_workspace_size, input->data_type):
                                nullptr;
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(dev_cuda->handle,
            &alpha,
            (const cudnnTensorDescriptor_t)input->desc, input->mem,
            (const cudnnTensorDescriptor_t)output_grad->desc, output_grad->mem,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            bwd_filter_algo, bwd_filter_workspace_mem, bwd_filter_workspace_size,
            &beta,
            (const cudnnFilterDescriptor_t)filter_desc, filter_grad->mem));
}
void op_convolution_cudnn::backward(){
    this->backward_data();
    this->backward_filter();
}
