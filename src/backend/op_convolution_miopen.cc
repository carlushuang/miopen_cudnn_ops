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
static const char * to_miopen_bwd_weights_algo_name(miopenConvBwdWeightsAlgorithm_t bwd_weight_algo){
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
void op_convolution_miopen::tune_op(){
    // API like miopenFindConvolutionForwardAlgorithm() need pre-alloced device memory
    // unlike nv cudnnFindConvolutionForwardAlgorithm()
    assert(input && output && filter);
    this->alloc_mem();
    if(!forward_tuned){
        forward_tuned=1;
#if 0
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
#endif

        CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(dev_hip->handle,
                (const miopenTensorDescriptor_t)filter->desc,
                (const miopenTensorDescriptor_t)input->desc,
                (const miopenConvolutionDescriptor_t )conv_desc->desc,
                (const miopenTensorDescriptor_t)output->desc,
                &fwd_workspace_size));
        //std::cout<<" -- request workspace size:"<<fwd_workspace_size<<std::endl;
        fwd_workspace_mem = fwd_workspace_size?
                        dev->ws->get(fwd_workspace_size, input->data_type):
                        nullptr;
        miopenConvAlgoPerf_t perfs[4];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionForwardAlgorithm(dev_hip->handle,
                (const miopenTensorDescriptor_t)input->desc, input->mem,
                (const miopenTensorDescriptor_t)filter->desc, filter->mem,
                (const miopenConvolutionDescriptor_t )conv_desc->desc,
                (const miopenTensorDescriptor_t)output->desc, output->mem,
                4, &returned_algos, perfs,
                fwd_workspace_mem, fwd_workspace_size, true));
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
    if(!(input_grad && output_grad && filter_grad))
        return ;        // ignore bwd
    if(!backward_data_tuned){
        backward_data_tuned = 1;
        CHECK_MIO(miopenConvolutionBackwardDataGetWorkSpaceSize(dev_hip->handle,
            (const miopenTensorDescriptor_t)output_grad->desc,
            (const miopenTensorDescriptor_t)filter->desc,
            (const miopenConvolutionDescriptor_t)conv_desc->desc,
            (const miopenTensorDescriptor_t)input_grad->desc,
            &bwd_data_workspace_size));
        bwd_data_workspace_mem = bwd_data_workspace_size?
                                dev->ws->get(bwd_data_workspace_size, input->data_type):
                                nullptr;
        
        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardDataAlgorithm(dev_hip->handle,
            (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
            (const miopenTensorDescriptor_t)filter->desc,filter->mem,
            (const miopenConvolutionDescriptor_t)conv_desc->desc,
            (const miopenTensorDescriptor_t)input_grad->desc, input_grad->mem,
            5, &returned_algos, perfs,
            bwd_data_workspace_mem, bwd_data_workspace_size, true));
#if 1
        LOG_I()<<" found miopenConv "<<returned_algos<<" bwd_data algo, using "<<perfs[0].bwd_data_algo<<"("<<
            to_miopen_bwd_data_algo_name(perfs[0].bwd_data_algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs[i].bwd_data_algo<< "(" <<to_miopen_bwd_data_algo_name(perfs[i].bwd_data_algo)
                 << ") - time: " << perfs[i].time << ", Memory: " << perfs[i].memory<<std::endl;
        }
#endif
        bwd_data_algo = perfs[0].bwd_data_algo;
    }
    if(!backward_filter_tuned){
        backward_filter_tune = 1;
        CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(dev_hip->handle,
            (const miopenTensorDescriptor_t)output_grad->desc,
            (const miopenTensorDescriptor_t)input->desc,
            (const miopenConvolutionDescriptor_t)conv_desc->desc,
            (const miopenTensorDescriptor_t)filter_grad->desc,
            &bwd_filter_workspace_size));
        bwd_filter_workspace_mem = bwd_filter_workspace_size?
                                dev->ws->get(bwd_filter_workspace_size, input->data_type):
                                nullptr;

        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardWeightsAlgorithm(dev_hip->handle,
            (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
            (const miopenTensorDescriptor_t)input->desc, input->mem,
            (const miopenConvolutionDescriptor_t)conv_desc->desc,
            (const miopenTensorDescriptor_t)filter_grad->desc, filter_grad->mem,
            5, &returned_algos, perfs,
            bwd_filter_workspace_mem, bwd_filter_workspace_size, true));
#if 1
        LOG_I()<<" found miopenConv "<<returned_algos<<" bwd_filter algo, using "<<perfs[0].bwd_weights_algo<<"("<<
            to_miopen_bwd_weights_algo_name(perfs[0].bwd_weights_algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs[i].bwd_weights_algo<< "(" <<to_miopen_bwd_weights_algo_name(perfs[i].bwd_weights_algo)
                 << ") - time: " << perfs[i].time << ", Memory: " << perfs[i].memory<<std::endl;
        }
#endif
        bwd_weights_algo = perfs[0].bwd_weights_algo;
    }
}

void op_convolution_miopen::forward(){
    assert(input && output && filter);
    device_hip * dev_hip = (device_hip *)dev;
    assert(forward_tuned);

    fwd_workspace_mem = fwd_workspace_size?
                        dev->ws->get(fwd_workspace_size, input->data_type):
                        nullptr;
    float alpha = 1.f;
    float beta = .0f;
    CHECK_MIO(miopenConvolutionForward(dev_hip->handle,
        &alpha,
        (const miopenTensorDescriptor_t)input->desc, input->mem,
        (const miopenTensorDescriptor_t)filter->desc, filter->mem,
        (const miopenConvolutionDescriptor_t )conv_desc->desc,
        fwd_algo,
        &beta,
        (const miopenTensorDescriptor_t)output->desc, output->mem,
        fwd_workspace_mem, fwd_workspace_size));
}
void op_convolution_miopen::backward_data(){
    assert(input && output && filter && input_grad && output_grad && filter_grad);
    device_hip * dev_hip = (device_hip *)dev;
    assert(backward_data_tuned);

    bwd_data_workspace_mem = bwd_data_workspace_size?
                                dev->ws->get(bwd_data_workspace_size, input->data_type):
                                nullptr;
    float alpha = 1.f;
    float beta = 0.f;
    CHECK_MIO(miopenConvolutionBackwardData(dev_hip->handle, &alpha,
        (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
        (const miopenTensorDescriptor_t)filter->desc,filter->mem,
        (const miopenConvolutionDescriptor_t)conv_desc->desc,
        bwd_data_algo, &beta,
        (const miopenTensorDescriptor_t)input_grad->desc, input_grad->mem,
        bwd_data_workspace_mem, bwd_data_workspace_size));
}
void op_convolution_miopen::backward_filter(){
    assert(input && output && filter && input_grad && output_grad && filter_grad);
    assert(backward_filter_tuned);
    device_hip * dev_hip = (device_hip *)dev;

    bwd_filter_workspace_mem = bwd_filter_workspace_size?
                                dev->ws->get(bwd_filter_workspace_size, input->data_type):
                                nullptr;
    float alpha = 1.f;
    float beta = 0.f;
    CHECK_MIO(miopenConvolutionBackwardWeights(dev_hip->handle, &alpha,
        (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
        (const miopenTensorDescriptor_t)input->desc, input->mem,
        (const miopenConvolutionDescriptor_t)conv_desc->desc,
        bwd_weights_algo, &beta,
        (const miopenTensorDescriptor_t)filter_grad->desc, filter_grad->mem,
        bwd_filter_workspace_mem, bwd_filter_workspace_size));
}
void op_convolution_miopen::backward(){
    this->backward_data();
    this->backward_filter();
}