#include "operator.hpp"

op_convolution_miopen::op_convolution_miopen(void * desc): op_convolution(desc){

}
op_convolution_miopen::~op_convolution_miopen(){

}
#define ALGO_CASE_STR(algo) \
    case algo: \
    return  #algo; \
    break

static const char * to_miopen_fwd_algo_name(miopenConvFwdAlgorithm_t fwd_algo){
    switch(fwd_algo){
        ALGO_CASE_STR(miopenConvolutionFwdAlgoGEMM);
        ALGO_CASE_STR(miopenConvolutionFwdAlgoDirect);
        ALGO_CASE_STR(miopenConvolutionFwdAlgoFFT);
        ALGO_CASE_STR(miopenConvolutionFwdAlgoWinograd);
        default:
            return "N/A";
        break;
    }
}
static const char * to_miopen_bwd_weight_algo_name(miopenConvBwdWeightsAlgorithm_t bwd_weight_algo){
    switch(bwd_weight_algo){
        ALGO_CASE_STR(miopenConvolutionBwdWeightsAlgoGEMM);
        ALGO_CASE_STR(miopenConvolutionBwdWeightsAlgoDirect);
        default:
            return "N/A";
        break;
    }
}

static const char * to_miopen_bwd_data_algo_name(miopenConvBwdDataAlgorithm_t bwd_data_algo){
    switch(bwd_data_algo){
        ALGO_CASE_STR(miopenConvolutionBwdDataAlgoGEMM);
        ALGO_CASE_STR(miopenConvolutionBwdDataAlgoDirect);
        ALGO_CASE_STR(miopenConvolutionBwdDataAlgoFFT);
        ALGO_CASE_STR(miopenConvolutionBwdDataAlgoWinograd);
        ALGO_CASE_STR(miopenTransposeBwdDataAlgoGEMM);
        default:
            return "N/A";
        break;
    }
}

void op_convolution_miopen::forward(){
    assert(input && output && filter);
    device_hip * dev_hip = (device_hip *)dev;
    if(!forward_prepared){
        forward_prepared=1;

        {
            int out_n, out_c, out_h, out_w;
            CHECK_MIO(miopenGetConvolutionForwardOutputDim((miopenConvolutionDescriptor_t )conv_desc->desc,
                (const miopenTensorDescriptor_t)input->desc,
                 (const miopenTensorDescriptor_t)filter->desc,
                 &out_n, &out_c, &out_h, &out_w));
            std::cout<<"-- expect output shape:"<<out_n<<"-"<<out_c<<"-"<<out_h<<"-"<<out_w<<std::endl;
        }

        dump_miopen_tensor_desc((const miopenTensorDescriptor_t)input->desc);
        dump_miopen_tensor_desc((const miopenTensorDescriptor_t)filter->desc);
        dump_miopen_tensor_desc((const miopenTensorDescriptor_t)output->desc);
        dump_miopen_convolution_desc((const miopenConvolutionDescriptor_t )conv_desc->desc);

        std::cout<<"["<<__func__<<"] "<<__LINE__<<std::endl;
        CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(dev_hip->handle,
                (const miopenTensorDescriptor_t)filter->desc,
                (const miopenTensorDescriptor_t)input->desc,
                (const miopenConvolutionDescriptor_t )conv_desc->desc,
                (const miopenTensorDescriptor_t)output->desc,
                &fwd_workspace_size));
        std::cout<<" -- request workspace size:"<<fwd_workspace_size<<std::endl;
        fwd_workspace_mem = fwd_workspace_size?
                        dev->ws->get(fwd_workspace_size, input->data_type):
                        nullptr;
        std::cout<<"["<<__func__<<"] "<<__LINE__<<std::endl;
        miopenConvAlgoPerf_t perfs[4];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionForwardAlgorithm(dev_hip->handle,
                (const miopenTensorDescriptor_t)input->desc, input->mem,
                (const miopenTensorDescriptor_t)filter->desc, filter->mem,
                (const miopenConvolutionDescriptor_t )conv_desc->desc,
                (const miopenTensorDescriptor_t)output->desc, output->mem,
                4, &returned_algos, perfs,
                fwd_workspace_mem, fwd_workspace_size, false));
        std::cout<<"["<<__func__<<"] "<<__LINE__<<std::endl;
#if 1
        LOG_I()<<" found miopenConv "<<returned_algos<<" fwd algo, using "<<perfs[0].fwd_algo<<"("<<
            to_miopen_fwd_algo_name(perfs[0].fwd_algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs[i].fwd_algo<< "(" <<to_miopen_fwd_algo_name(perfs[i].fwd_algo)
                 << ") - time: " << perfs[i].time << ", Memory: " << perfs[i].memory<<std::endl;
        }
#endif
        fwd_algo = perfs[0].fwd_algo;
        fwd_workspace_size = perfs[0].memory;   // wrap back the selected algo size
    }

    fwd_workspace_mem = fwd_workspace_size?
                        dev->ws->get(fwd_workspace_size, input->data_type):
                        nullptr;
    float alpha = 1.f;
    float beta = .0f;
    std::cout<<"["<<__func__<<"] "<<__LINE__<<std::endl;
    CHECK_MIO(miopenConvolutionForward(dev_hip->handle,
        &alpha,
        (const miopenTensorDescriptor_t)input->desc, input->mem,
        (const miopenTensorDescriptor_t)filter->desc, filter->mem,
        (const miopenConvolutionDescriptor_t )conv_desc->desc,
        fwd_algo,
        &beta,
        (const miopenTensorDescriptor_t)output->desc, output->mem,
        fwd_workspace_mem, fwd_workspace_size));
    std::cout<<"["<<__func__<<"] "<<__LINE__<<std::endl;
}
void op_convolution_miopen::backward(){

}