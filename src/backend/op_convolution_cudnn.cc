#include "operator.hpp"

op_convolution_cudnn::op_convolution_cudnn(void * desc) : op_convolution(desc){

}
op_convolution_cudnn::~op_convolution_cudnn(){
    if(forward_tuned){
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
void op_convolution_cudnn::tune_op(){
    assert(input && output && filter);
    device_cuda * dev_cuda = (device_cuda *)dev;
    if(!forward_tuned){
        forward_tuned = 1;
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
		/*
        CHECK_CUDNN(cudnnSetConvolutionMathType((const cudnnConvolutionDescriptor_t)conv_desc->desc,
                CUDNN_DEFAULT_MATH));
				*/

        // find fwd algo
#if 0
        cudnnConvolutionFwdAlgoPerf_t perfs[8];
        int returned_algos;

        CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(dev_cuda->handle,
            (const cudnnTensorDescriptor_t)input->desc,
            filter_desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)output->desc,
            8, &returned_algos, perfs));

#ifdef OP_VERBOSE
        LOG_I()<<" found cudnnConv "<<returned_algos<<" fwd algo, using "<<perfs[0].algo<<"("<<
            to_cudnn_fwd_algo_name(perfs[0].algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs[i].algo<< "(" <<to_cudnn_fwd_algo_name(perfs[i].algo)
                 << ") - time: " << perfs[i].time << ", Memory: " << perfs[i].memory<<std::endl;
        }
#endif
        fwd_algo = perfs[0].algo;
#ifdef OP_CONV_SELECT
        for(int i=0;i<returned_algos;i++){
            if(perfs[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING || 
                perfs[i].algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT){
                fwd_algo = perfs[i].algo;
                break;
            }
        }
#endif
#else
        CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(dev_cuda->handle,
            (const cudnnTensorDescriptor_t)input->desc,
            filter_desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)output->desc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwd_algo));
#endif

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

    if(!(input_grad && output_grad))
        return ;        // ignore bwd
    if(!backward_data_tuned){
        backward_data_tuned = 1;

        // find bwd data algo
        cudnnConvolutionBwdDataAlgoPerf_t perfs_data[5];
        int returned_algos;
        CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(dev_cuda->handle, 
            filter_desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)input_grad->desc,
            5, &returned_algos, perfs_data));

#ifdef OP_VERBOSE
        LOG_I()<<" found cudnnConv "<<returned_algos<<" bwd_data algo, using "<<perfs_data[0].algo<<"("<<
            to_cudnn_bwd_data_algo_name(perfs_data[0].algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs_data[i].algo<< "(" <<to_cudnn_bwd_data_algo_name(perfs_data[i].algo)
                 << ") - time: " << perfs_data[i].time << ", Memory: " << perfs_data[i].memory<<std::endl;
        }
#endif
        bwd_data_algo = perfs_data[0].algo;

        CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(dev_cuda->handle,
            filter_desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            (const cudnnTensorDescriptor_t)input_grad->desc,
            bwd_data_algo, &bwd_data_workspace_size));
        //bwd_data_workspace_mem = bwd_data_workspace_size?
        //                        dev->ws->get(bwd_data_workspace_size, input->data_type):
        //                        nullptr;
    }

	if (!filter_grad)
		return;

    if(!backward_filter_tuned){
        backward_filter_tuned = 1;

        // find bwd filter algo
        cudnnConvolutionBwdFilterAlgoPerf_t perfs_filter[5];
        int returned_algos;
        CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(dev_cuda->handle, 
            (const cudnnTensorDescriptor_t)input->desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            filter_desc,
            5, &returned_algos, perfs_filter));
#ifdef OP_VERBOSE
        LOG_I()<<" found cudnnConv "<<returned_algos<<" bwd_filter algo, using "<<perfs_filter[0].algo<<"("<<
            to_cudnn_bwd_filter_algo_name(perfs_filter[0].algo)<<")"<<std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            LOG_I()<<"    " << i << ": " << perfs_filter[i].algo<< "(" <<to_cudnn_bwd_filter_algo_name(perfs_filter[i].algo)
                 << ") - time: " << perfs_filter[i].time << ", Memory: " << perfs_filter[i].memory<<std::endl;
        }
#endif
        bwd_filter_algo = perfs_filter[0].algo;

        CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(dev_cuda->handle,
            (const cudnnTensorDescriptor_t)input->desc,
            (const cudnnTensorDescriptor_t)output_grad->desc,
            (const cudnnConvolutionDescriptor_t)conv_desc->desc,
            filter_desc,
            bwd_filter_algo, &bwd_filter_workspace_size));
        //bwd_filter_workspace_mem = bwd_filter_workspace_size?
        //                        dev->ws->get(bwd_filter_workspace_size, input->data_type):
        //                        nullptr;
    }
}


void op_convolution_cudnn::forward(){
    assert(input && output && filter);
    assert(forward_tuned);
    device_cuda * dev_cuda = (device_cuda *)dev;

#if 0
    dump_cudnn_convolution_desc((const cudnnConvolutionDescriptor_t)conv_desc->desc);
    dump_cudnn_filter_desc(filter_desc);
    dump_cudnn_tensor_desc((const cudnnTensorDescriptor_t)input->desc);
    dump_cudnn_tensor_desc((const cudnnTensorDescriptor_t)filter->desc);
    dump_cudnn_tensor_desc((const cudnnTensorDescriptor_t)output->desc);
#endif
    float alpha = 1.0f;
    float beta = 0.0f;
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
    assert(filter && input_grad && output_grad);
    assert(backward_data_tuned);
    device_cuda * dev_cuda = (device_cuda *)dev;

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
    assert(backward_filter_tuned);
    device_cuda * dev_cuda = (device_cuda *)dev;

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
std::string op_convolution_cudnn::get_fwd_algo_name(){
    std::string algo_name(to_cudnn_fwd_algo_name(fwd_algo));
    return algo_name;
}
std::string op_convolution_cudnn::get_bwd_data_name(){
    std::string algo_name(to_cudnn_bwd_data_algo_name(bwd_data_algo));
    return algo_name;
}
std::string op_convolution_cudnn::get_bwd_filter_name(){
    std::string algo_name(to_cudnn_bwd_filter_algo_name(bwd_filter_algo));
    return algo_name;
}

void op_convolution_cudnn::print_fwd_time(const float kernel_average_time) {
	std::string fwd_algo_name = get_fwd_algo_name();
	std::cout << "OpDriver Forward Conv. Algorithm: " << fwd_algo_name << "." << std::endl;

	printf("GPU Kernel Time Forward Conv. Elapsed: %f ms (average)\n", kernel_average_time);
	int in_n, in_c, in_h, in_w;
	int wei_k, wei_c, wei_h, wei_w;
	int out_n, out_c, out_h, out_w;
	
    cudnnDataType_t dt;
    cudnnTensorFormat_t fmt;
    int n_stride, c_stride, h_stride, w_stride;

	CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)input->desc, &dt,
				&in_n, &in_c, &in_h, &in_w, &n_stride, &c_stride, &h_stride, &w_stride));
	CHECK_CUDNN(cudnnGetFilter4dDescriptor((cudnnFilterDescriptor_t)filter_desc, &dt,
				&fmt, &wei_k, &wei_c, &wei_h, &wei_w));
	CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)output->desc, &dt,
				&out_n, &out_c, &out_h, &out_w, &n_stride, &c_stride, &h_stride, &w_stride));

	debug_msg("input:(%d,%d,%d,%d), filer:(%d,%d,%d,%d), output:(%d,%d,%d,%d)\n",
			in_n, in_c, in_h, in_w, wei_k, wei_c, wei_h, wei_w, out_n, out_c, out_h, out_w);

	size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w;
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
}

void op_convolution_cudnn::print_bwd_time(const float kernel_average_time) {
	std::string algo_name = get_bwd_data_name();
	std::cout << "OpDriver Backward Data Conv. Algorithm: " << algo_name << "." << std::endl;

	printf("GPU Kernel Time Backward Data Conv. Elapsed: %f ms (average)\n", kernel_average_time);
	int in_n, in_c, in_h, in_w;
	int wei_k, wei_c, wei_h, wei_w;
	int out_n, out_c, out_h, out_w;
	
    cudnnDataType_t dt;
    cudnnTensorFormat_t fmt;
    int n_stride, c_stride, h_stride, w_stride;

	CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)input->desc, &dt,
				&in_n, &in_c, &in_h, &in_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_CUDNN(cudnnGetFilter4dDescriptor((cudnnFilterDescriptor_t)filter_desc, &dt,
				&fmt, &wei_k, &wei_c, &wei_h, &wei_w));
	CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)output->desc, &dt,
				&out_n, &out_c, &out_h, &out_w, &n_stride, &c_stride, &h_stride,
				&w_stride));

	debug_msg("input:(%d,%d,%d,%d), filer:(%d,%d,%d,%d), output:(%d,%d,%d,%d)\n",
			in_n, in_c, in_h, in_w, wei_k, wei_c, wei_h, wei_w, out_n, out_c, out_h, out_w);

	size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w;
	size_t inBytes = in_n * in_c * in_h * in_w * 4;
	size_t weiBytes = wei_k * wei_c * wei_h * wei_w * 4;
	size_t readBytes = inBytes + weiBytes;
	size_t outputBytes = out_n * out_c * out_h * out_w * 4;

	printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
			   "GB/s, timeMs\n");
	printf("stats: %s%dx%d, %u, %u, %u, %u, %u, %u, %u, %zu, %zu, %zu, %.0f, %.0f, %f\n",
		   "bwdd-conv",
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
}

void op_convolution_cudnn::print_wrw_time(const float kernel_average_time) {
	std::string algo_name = get_bwd_filter_name();
	std::cout << "OpDriver Backward Weights Conv. Algorithm: " << algo_name << "." << std::endl;

	printf("GPU Kernel Time Backward Weights Conv. Elapsed: %f ms (average)\n", kernel_average_time);
	int in_n, in_c, in_h, in_w;
	int wei_k, wei_c, wei_h, wei_w;
	int out_n, out_c, out_h, out_w;
	
    cudnnDataType_t dt;
    cudnnTensorFormat_t fmt;
    int n_stride, c_stride, h_stride, w_stride;

	CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)input->desc, &dt,
				&in_n, &in_c, &in_h, &in_w, &n_stride, &c_stride, &h_stride,
				&w_stride));
	CHECK_CUDNN(cudnnGetFilter4dDescriptor((cudnnFilterDescriptor_t)filter_desc, &dt,
				&fmt, &wei_k, &wei_c, &wei_h, &wei_w));
	CHECK_CUDNN(cudnnGetTensor4dDescriptor((cudnnTensorDescriptor_t)output->desc, &dt,
				&out_n, &out_c, &out_h, &out_w, &n_stride, &c_stride, &h_stride,
				&w_stride));

	debug_msg("input:(%d,%d,%d,%d), filer:(%d,%d,%d,%d), output:(%d,%d,%d,%d)\n",
			in_n, in_c, in_h, in_w, wei_k, wei_c, wei_h, wei_w, out_n, out_c, out_h, out_w);

	size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w;
	size_t readBytes = 0;
	size_t outputBytes = 0;

	printf("stats: name, n, c, ho, wo, x, y, k, flopCnt, bytesRead, bytesWritten, GFLOPs, "
			   "GB/s, timeMs\n");
	printf("stats: %s%dx%d, %u, %u, %u, %u, %u, %u, %u, %zu, %zu, %zu, %.0f, %.0f, %f\n",
		   "bwdw-conv",
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
}
