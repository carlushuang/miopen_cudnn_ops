#include "operator.hpp"
#include "backend.hpp"

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
        ALGO_CASE_STR(miopenConvolutionBwdWeightsAlgoWinograd);
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

#ifdef OP_LOG_TO_FILE
static inline FILE * get_fp(){
    static int inited = 0;
    FILE * fp;
    fp = fopen ("conv_log.csv","a");
    if(fp && !inited){
        inited = 1;
        FILE * fp_2 = fopen("conv_log_banner.csv","w");
        if(fp_2){
            fprintf(fp_2, "n,g,c,h,w,k,y,x,ho,wo,py,px,sy,sx,dy,dx,gflop");
            fprintf(fp_2, ",algo(fwd),workspace,time(ms),gflops,efficiency(%)");
            fprintf(fp_2, ",algo(bwd),workspace,time(ms),gflops,efficiency(%)");
            fprintf(fp_2, ",algo(wrw),workspace,time(ms),gflops,efficiency(%)");
            fprintf(fp_2, "\n");
            fclose(fp_2);
        }
    }
    return fp;
}
#endif

#define MIOPEN_EXHAUSTIVE_SEARCH false
void op_convolution_miopen::tune_op(){
    // API like miopenFindConvolutionForwardAlgorithm() need pre-alloced device memory
    // unlike nv cudnnFindConvolutionForwardAlgorithm()
    this->alloc_mem();

    device_hip * dev_hip = (device_hip *)dev;
    assert(input && output && filter);

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
                fwd_workspace_mem, fwd_workspace_size, MIOPEN_EXHAUSTIVE_SEARCH));
#ifdef OP_VERBOSE
        LOG_I()<<" found miopenConv "<<returned_algos<<" fwd algo, using "<<perfs[0].fwd_algo<<"("<<
            to_miopen_fwd_algo_name(perfs[0].fwd_algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs[i].fwd_algo<< "(" <<to_miopen_fwd_algo_name(perfs[i].fwd_algo)
                 << ") - time: " << perfs[i].time << ", Memory: " << perfs[i].memory<<std::endl;
        }
#endif

        fwd_algo = perfs[0].fwd_algo;
        fwd_workspace_size = perfs[0].memory;   // wrap back the selected algo size
#ifdef OP_CONV_SELECT
        for(int i=0;i<returned_algos;i++){
            if(perfs[i].fwd_algo == miopenConvolutionFwdAlgoFFT){
                fwd_algo = perfs[i].fwd_algo;
                fwd_workspace_size = perfs[i].memory;
                break;
            }
        }
#endif
    }
    if(!(input_grad && output_grad))
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
            bwd_data_workspace_mem, bwd_data_workspace_size, MIOPEN_EXHAUSTIVE_SEARCH));
#ifdef OP_VERBOSE
        LOG_I()<<" found miopenConv "<<returned_algos<<" bwd_data algo, using "<<perfs[0].bwd_data_algo<<"("<<
            to_miopen_bwd_data_algo_name(perfs[0].bwd_data_algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs[i].bwd_data_algo<< "(" <<to_miopen_bwd_data_algo_name(perfs[i].bwd_data_algo)
                 << ") - time: " << perfs[i].time << ", Memory: " << perfs[i].memory<<std::endl;
        }
#endif
        bwd_data_algo = perfs[0].bwd_data_algo;
    }

    if(!filter_grad)
        return;        // ignore wrw

    if(!backward_filter_tuned){
        backward_filter_tuned = 1;
        CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(dev_hip->handle,
            (const miopenTensorDescriptor_t)output_grad->desc,
            (const miopenTensorDescriptor_t)input->desc,
            (const miopenConvolutionDescriptor_t)conv_desc->desc,
            (const miopenTensorDescriptor_t)filter_grad->desc,
            &bwd_filter_workspace_size));
        bwd_filter_workspace_mem = bwd_filter_workspace_size?
                                dev->ws->get(bwd_filter_workspace_size, input_grad->data_type):
                                nullptr;

        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardWeightsAlgorithm(dev_hip->handle,
            (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
            (const miopenTensorDescriptor_t)input->desc, input->mem,
            (const miopenConvolutionDescriptor_t)conv_desc->desc,
            (const miopenTensorDescriptor_t)filter_grad->desc, filter_grad->mem,
            5, &returned_algos, perfs,
            bwd_filter_workspace_mem, bwd_filter_workspace_size, MIOPEN_EXHAUSTIVE_SEARCH));
#ifdef OP_VERBOSE
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
    assert(filter && input_grad && output_grad);
    device_hip * dev_hip = (device_hip *)dev;
    assert(backward_data_tuned);
	// assert(bwd_data_workspace_mem);

    bwd_data_workspace_mem = bwd_data_workspace_size?
                                dev->ws->get(bwd_data_workspace_size, input->data_type):
                                nullptr;

    const float alpha = 1.f;
    const float beta = 0.f;
    CHECK_MIO(miopenConvolutionBackwardData(dev_hip->handle, &alpha,
        (const miopenTensorDescriptor_t)output_grad->desc, output_grad->mem,
        (const miopenTensorDescriptor_t)filter->desc,filter->mem,
        (const miopenConvolutionDescriptor_t)conv_desc->desc,
        bwd_data_algo, &beta,
        (const miopenTensorDescriptor_t)input_grad->desc, input_grad->mem,
        bwd_data_workspace_mem, bwd_data_workspace_size));
}
void op_convolution_miopen::backward_filter(){
    assert(input_grad && output_grad && filter_grad);
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
std::string op_convolution_miopen::get_fwd_algo_name(){
    std::string algo_name(to_miopen_fwd_algo_name(fwd_algo));
    return algo_name;
}
std::string op_convolution_miopen::get_bwd_data_name(){
    std::string algo_name(to_miopen_bwd_data_algo_name(bwd_data_algo));
    return algo_name;
}
std::string op_convolution_miopen::get_bwd_filter_name(){
    std::string algo_name(to_miopen_bwd_weights_algo_name(bwd_weights_algo));
    return algo_name;
}

void op_convolution_miopen::print_fwd_time(const float kernel_average_time) {
	std::string fwd_algo_name = get_fwd_algo_name();
	std::cout << "OpDriver Forward Conv. Algorithm: " << fwd_algo_name << "." << std::endl;

	printf("GPU Kernel Time Forward Conv. Elapsed: %f ms (average)\n", kernel_average_time);
	int in_n, in_c, in_h, in_w;
	int wei_k, wei_c, wei_h, wei_w;
	int out_n, out_c, out_h, out_w;
	
    miopenDataType_t dt;
    int n_stride, c_stride, h_stride, w_stride;

	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)input->desc, &dt,
				&in_n, &in_c, &in_h, &in_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)filter->desc, &dt,
				&wei_k, &wei_c, &wei_h, &wei_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)output->desc, &dt,
				&out_n, &out_c, &out_h, &out_w, &n_stride, &c_stride, &h_stride,
				&w_stride));

	debug_msg("input:(%d,%d,%d,%d), filer:(%d,%d,%d,%d), output:(%d,%d,%d,%d)\n",
			in_n, in_c, in_h, in_w, wei_k, wei_c, wei_h, wei_w, out_n, out_c, out_h, out_w);

	size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w / conv_desc->groups;
	size_t inBytes = in_n * in_c * in_h * in_w * 4;
	size_t weiBytes = wei_k * wei_c * wei_h * wei_w * 4;
	size_t readBytes = inBytes + weiBytes;
	size_t outputBytes = out_n * out_c * out_h * out_w * 4;

	printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
			   "GB/s, timeMs\n");
	printf("stats: %s%dx%d, %u, %u, %u, %u, %u, %u, %u, %zu, %zu, %zu, %.0f, %.0f, %f\n",
		   "fwd-conv",
		   wei_h,
		   wei_w,
		   in_n,
		   in_c,
		   out_h,
		   out_w,
		   wei_h,
		   wei_w,
		   out_c,
		   flopCnt,
		   readBytes,
		   outputBytes,
		   flopCnt / kernel_average_time / 1e6,
		   (readBytes + outputBytes) / kernel_average_time / 1e6,
		   kernel_average_time);
#ifdef OP_LOG_TO_FILE
    {
        FILE * fp = get_fp();
        if(fp){
                        // n,g,c,h,w,k,y,x,ho,wo,py,px,sy,sx,dy,dx,gflop
            fprintf(fp, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.2f",
                in_n, conv_desc->groups, in_c, in_h, in_w, wei_k, wei_h, wei_w, out_h, out_w,
                conv_desc->padding[0],conv_desc->padding[1],
                conv_desc->stride[0], conv_desc->stride[1], conv_desc->dilation[0], conv_desc->dilation[1],
                (double)flopCnt/1e9);
                    // ,algo(fwd),workspace,time(ms),gflops,efficiency(%)");
            fprintf(fp, ",%s,%s,%.2f,%.2f,%.2f%%",
                fwd_algo_name.c_str(), util_b2string(fwd_workspace_size).c_str(),
                kernel_average_time, flopCnt / kernel_average_time / 1e6,
                flopCnt / kernel_average_time / 1e4 / dev->get_theoretical_gflops(input->data_type));
            fclose(fp);
        }
    }
#endif
}

void op_convolution_miopen::print_bwd_time(const float kernel_average_time) {
	std::string algo_name = get_bwd_data_name();
	std::cout << "OpDriver Backward Data Conv. Algorithm: " << algo_name << "." << std::endl;

	printf("GPU Kernel Time Backward Data Conv. Elapsed: %f ms (average)\n", kernel_average_time);
	int in_n, in_c, in_h, in_w;
	int wei_k, wei_c, wei_h, wei_w;
	int out_n, out_c, out_h, out_w;
	
    miopenDataType_t dt;
    int n_stride, c_stride, h_stride, w_stride;

	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)input->desc, &dt,
				&in_n, &in_c, &in_h, &in_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)filter->desc, &dt,
				&wei_k, &wei_c, &wei_h, &wei_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)output->desc, &dt,
				&out_n, &out_c, &out_h, &out_w, &n_stride, &c_stride, &h_stride,
				&w_stride));

	debug_msg("input:(%d,%d,%d,%d), filer:(%d,%d,%d,%d), output:(%d,%d,%d,%d)\n",
			in_n, in_c, in_h, in_w, wei_k, wei_c, wei_h, wei_w, out_n, out_c, out_h, out_w);

	size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w / conv_desc->groups;
	size_t inBytes = in_n * in_c * in_h * in_w * 4;
	size_t weiBytes = wei_k * wei_c * wei_h * wei_w * 4;
	size_t readBytes = inBytes + weiBytes;
	size_t outputBytes = out_n * out_c * out_h * out_w * 4;

	printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
			   "GB/s, timeMs\n");
	printf("stats: %s%dx%d, %u, %u, %u, %u, %u, %u, %u, %zu, %zu, %zu, %.0f, %.0f, %f\n",
		   "fwd-conv",
		   wei_h,
		   wei_w,
		   in_n,
		   in_c,
		   out_h,
		   out_w,
		   wei_h,
		   wei_w,
		   out_c,
		   flopCnt,
		   readBytes,
		   outputBytes,
		   flopCnt / kernel_average_time / 1e6,
		   (readBytes + outputBytes) / kernel_average_time / 1e6,
		   kernel_average_time);
#ifdef OP_LOG_TO_FILE
    {
        FILE * fp = get_fp();
        if(fp){
                    // ,algo(fwd),workspace,time(ms),gflops,efficiency(%)");
            fprintf(fp, ",%s,%s,%.2f,%.2f,%.2f%%",
                algo_name.c_str(), util_b2string(bwd_data_workspace_size).c_str(),
                kernel_average_time, flopCnt / kernel_average_time / 1e6,
                flopCnt / kernel_average_time / 1e4 / dev->get_theoretical_gflops(input->data_type));
            fclose(fp);
        }
    }
#endif
}

void op_convolution_miopen::print_wrw_time(const float kernel_average_time) {
	std::string algo_name = get_bwd_filter_name();
	std::cout << "OpDriver Backward Weights Conv. Algorithm: " << algo_name << "." << std::endl;

	printf("GPU Kernel Time Backward Weights Conv. Elapsed: %f ms (average)\n", kernel_average_time);
	int in_n, in_c, in_h, in_w;
	int wei_k, wei_c, wei_h, wei_w;
	int out_n, out_c, out_h, out_w;
	
    miopenDataType_t dt;
    int n_stride, c_stride, h_stride, w_stride;

	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)input->desc, &dt,
				&in_n, &in_c, &in_h, &in_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)filter->desc, &dt,
				&wei_k, &wei_c, &wei_h, &wei_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_MIO(miopenGet4dTensorDescriptor((miopenTensorDescriptor_t)output->desc, &dt,
				&out_n, &out_c, &out_h, &out_w, &n_stride, &c_stride, &h_stride,
				&w_stride));

	debug_msg("input:(%d,%d,%d,%d), filer:(%d,%d,%d,%d), output:(%d,%d,%d,%d)\n",
			in_n, in_c, in_h, in_w, wei_k, wei_c, wei_h, wei_w, out_n, out_c, out_h, out_w);

	size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w / conv_desc->groups;
	size_t readBytes = 0;
	size_t outputBytes = 0;

	printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
			   "GB/s, timeMs\n");
	printf("stats: %s%dx%d, %u, %u, %u, %u, %u, %u, %u, %zu, %zu, %zu, %.0f, %.0f, %f\n",
		   "fwd-conv",
		   wei_h,
		   wei_w,
		   in_n,
		   in_c,
		   out_h,
		   out_w,
		   wei_h,
		   wei_w,
		   out_c,
		   flopCnt,
		   readBytes,
		   outputBytes,
		   flopCnt / kernel_average_time / 1e6,
		   (readBytes + outputBytes) / kernel_average_time / 1e6,
		   kernel_average_time);
#ifdef OP_LOG_TO_FILE
    {
        FILE * fp = get_fp();
        if(fp){
                    // ,algo(fwd),workspace,time(ms),gflops,efficiency(%)");
            fprintf(fp, ",%s,%s,%.2f,%.2f,%.2f%%",
                algo_name.c_str(), util_b2string(bwd_filter_workspace_size).c_str(),
                kernel_average_time, flopCnt / kernel_average_time / 1e6,
                flopCnt / kernel_average_time / 1e4 / dev->get_theoretical_gflops(input->data_type));
            fprintf(fp,"\n");
            fclose(fp);
        }
    }
#endif
}
