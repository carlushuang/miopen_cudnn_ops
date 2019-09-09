#include "backend.hpp"
#include "operator.hpp"
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <unordered_map>
#include <random>
#include <fstream>

#include <unistd.h>

#define EF_PRT

#define LOOP_WARMUP 2
#define LOOP_ITR    5

static device_base * determin_device(){
	std::string backend;
#ifdef WITH_MIOPEN
	backend = "HIP";
#endif
#ifdef WITH_CUDNN
	backend = "CUDA";
#endif
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
void inline b2s(size_t bytes, char * str){
	if(bytes<1024){
		sprintf(str, "%lluB", bytes);
	}else if(bytes<(1024*1024)){
		double b= (double)bytes/1024.0;
		sprintf(str, "%.2fKB", b);
	}else if(bytes<(1024*1024*1024)){
		double b= (double)bytes/(1024.0*1024);
		sprintf(str, "%.2fMB", b);
	}else{
		double b= (double)bytes/(1024.0*1024*1024);
		sprintf(str, "%.2fGB", b);
	}
}

static void rand_float(float * vec, int len){
	static std::random_device rd;
	static std::mt19937 mt(rd());
	static std::uniform_real_distribution<float> dist(-1.f, 1.f);

	for(int i=0;i<len;i++){
		vec[i] = dist(mt);
	}
}

// parse int arg value
class arg_parser{
#define ARG_VALUE_INIT "n/a"
	public:
		struct arg_store{
			std::string arg_name;
			std::string value;
			std::string default_value;
			std::string help_str;
		};
		arg_parser(const char * _name):name(_name){};
		~arg_parser(){}
		void insert_arg(const char *long_name, const char arg, std::string default_value, const char * help, const std::string &type){
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
				if(arg_name == "--help" || arg_name == "-help"){
					usage();
					return false;
				}
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
				arg_pair[arg_name].value = argv[i+1];
			}
			return true;
		}
		std::string get_arg(const char * arg){
			std::string arg_name = std::string("-") + arg;
			if(arg_pair.count(arg_name) == 0){
				std::cerr<<"no such arg "<<arg_name<<std::endl;;
				usage();
				assert(0 && "no such arg in parse arg");
			}
			std::string val = arg_pair[arg_name].value;
			if(val == ARG_VALUE_INIT)
				val = arg_pair[arg_name].default_value;
			return val; 
		}
		int  get_arg_int(const char * arg){
			std::string val = get_arg(arg);
			return atoi(val.c_str());
		}
		float get_arg_float(const char * arg){
			std::string val = get_arg(arg);
			return (float)atof(val.c_str());
		}
		void usage(){
			std::cout<<name<<" args:"<<std::endl;
			//std::cout<<"    --help, print usage and return"<<std::endl;
			for(auto & it : arg_pair){
				arg_store a = it.second;
				std::cout<<"    "<<it.first<<", "<<a.help_str<<
					"(default:"<<a.default_value<<")"
#if 0
					<<", cur:"<<a.value
#endif           
					<<std::endl;
			}
		}
		void dump_parsed(){
			std::cout<<"using args: "<<name<<" ";
			for(auto & it : arg_pair){
				arg_store a = it.second;
				std::cout<<" "<<it.first<<" "<<(a.value==ARG_VALUE_INIT?a.default_value:a.value);
			}
			std::cout<<std::endl;
		}
	private:
		std::string name;
		std::unordered_map<std::string, arg_store> arg_pair;
};

static device_base * gpu_dev;
static device_base * cpu_dev;

template <typename T>
void writeToTxt(const char* fileName, T* data, size_t dataNumItems)
{
	std::ofstream outFile(fileName, std::ios::binary);

	if(outFile)
	{    
		for (size_t i = 0; i < dataNumItems; i++)
			outFile << std::setprecision(9) << data[i] << ' ';
		outFile << std::endl;
		outFile.close();
		debug_msg("Wrote output to file %s\n", fileName);
	}    
	else 
	{    
		debug_msg("Could not open file %s for writing\n", fileName);
	}    
}

template <typename T>
bool readFromTxt(T *data, size_t dataNumItems, const char* fileName)
{
	std::ifstream infile(fileName, std::ios::binary);

	if(infile)
	{    
		for (size_t i = 0; i < dataNumItems; i++)
			infile >> data[i];
		infile.close();
		debug_msg("Read data from input file %s\n", fileName);
		return true;
	}    
	else 
	{    
		debug_msg("Could not open file %s for reading\n", fileName);
		return false;
	}    
}

#if 0
static int pooling_driver(int argc, char ** argv){
	arg_parser parser("pooling");
	parser.insert_arg("k", "kernel size", "2");
	parser.insert_arg("s", "stride", "2");
	parser.insert_arg("p", "padding", "0");
	parser.insert_arg("n", "batch", "2");
	parser.insert_arg("c", "channel", "3");
	parser.insert_arg("h", "height", "128");
	parser.insert_arg("w", "width", "128");
	parser.insert_arg("m", "pooling mode, "
			"0-MAX 1-MAX_DETERMINISTIC "
			"2-AVG_EXCLUSIVE 3-AVG_INCLUSIVE ", "0");
	parser.insert_arg("f", "forward(1) or backward(0)", "0");

	// parse arg
	if(!parser.parse(argc, argv)) return -1;
	parser.dump_parsed();

	int ksize = parser.get_arg_int("k");
	int psize = parser.get_arg_int("p");
	int ssize = parser.get_arg_int("s");
	size_t n  = (size_t)parser.get_arg_int("n");
	size_t c  = (size_t)parser.get_arg_int("c");
	size_t h  = (size_t)parser.get_arg_int("h");
	size_t w  = (size_t)parser.get_arg_int("w");
	int pmode = parser.get_arg_int("m");
	int is_fwd = parser.get_arg_int("f");

	// get param from arg
	int pooling_kernel[2] = {ksize,ksize};
	int pooling_stride[2] = {ssize,ssize};
	int pooling_padding[2] = {psize,psize};
	pooling_mode pm = POOLING_MAX;
	if(pmode == 0) pm = POOLING_MAX;
	else if(pmode == 1) pm = POOLING_MAX_DETERMINISTIC;
	else if(pmode == 2) pm = POOLING_AVG_EXCLUSIVE;
	else if(pmode == 3) pm = POOLING_AVG_INCLUSIVE;
	else {std::cout<<"unsupport pooing mode "<<pmode<<std::endl; return -1;}

	// start
	pooling_desc_t * pooling_desc = gpu_dev->pooling_desc_create(
			pooling_kernel, pooling_stride, pooling_padding, 2,
			pm);
	pooling_desc_t * pooling_desc_c = cpu_dev->pooling_desc_create(
			pooling_kernel, pooling_stride, pooling_padding, 2,
			pm);
	tensor_t *t_in, *t_out, *t_in_c, *t_out_c;
	tensor_t *t_in_grad, *t_out_grad, *t_in_grad_c, *t_out_grad_c;
	operator_base *op_pooling, *op_pooling_c;
	size_t t_in_dim[4] = {n,c,h,w};
	size_t t_out_dim[4];

	// create gpu tensors
	op_pooling = operator_create(gpu_dev, OP_POOLING, pooling_desc);

	t_in = gpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_pooling->input = t_in;
	op_pooling->infer_shape(t_out_dim);
	t_out = gpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_pooling->output = t_out;
	if(!is_fwd){
		t_in_grad = gpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		t_out_grad = gpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		op_pooling->input_grad = t_in_grad;
		op_pooling->output_grad = t_out_grad;
	}

	// create cpu tensors
	op_pooling_c = operator_create(cpu_dev, OP_POOLING, pooling_desc_c);
	t_in_c = cpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	t_out_c = cpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_pooling_c->input = t_in_c;
	op_pooling_c->output = t_out_c;
	if(!is_fwd){
		t_in_grad_c = cpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		t_out_grad_c = cpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		op_pooling_c->input_grad = t_in_grad_c;
		op_pooling_c->output_grad = t_out_grad_c;
	}

	op_pooling->tune_op();
	op_pooling->alloc_mem();
	op_pooling_c->tune_op();
	op_pooling_c->alloc_mem();

	// prepare input
	rand_float((float*)t_in_c->mem, t_in_c->elem());
	gpu_dev->tensor_copy(t_in, t_in_c->mem, t_in_c->bytes(), TENSOR_COPY_H2D);
	if(!is_fwd){
		rand_float((float*)t_out_grad_c->mem, t_out_grad_c->elem());
		gpu_dev->tensor_copy(t_out_grad, t_out_grad_c->mem, t_out_grad_c->bytes(), TENSOR_COPY_H2D);

		cpu_dev->tensor_set(t_in_grad_c, 0);
		gpu_dev->tensor_set(t_in_grad, 0);
	}

	op_pooling->forward();
	if(!is_fwd)
		op_pooling->backward();
	//validation
	op_pooling_c->forward();
	if(!is_fwd)
		op_pooling_c->backward();

	// compare
	float * dev_out = new float[t_out->elem()];
	gpu_dev->tensor_copy(dev_out, t_out, t_out->bytes(), TENSOR_COPY_D2H);

	int error_cnt = util_compare_data(dev_out, t_out_c->mem, t_out_c->elem(), TENSOR_DT_FLOAT, 0.001);
	if(error_cnt){
		std::cout<<"pooling fwd compare fail"<<std::endl;
	}else{
		std::cout<<"pooling fwd result verified"<<std::endl;
	}
	delete [] dev_out;
	if(!is_fwd){
		float * dev_in_grad = new float[t_in_grad->elem()];
		gpu_dev->tensor_copy(dev_in_grad, t_in_grad, t_in_grad->bytes(), TENSOR_COPY_D2H);

		int error_cnt_grad = util_compare_data(dev_in_grad, t_in_grad_c->mem, t_in_grad_c->elem(), TENSOR_DT_FLOAT, 0.001);
		if(error_cnt_grad){
			std::cout<<"pooling bwd compare fail"<<std::endl;
		}else{
			std::cout<<"pooling bwd result verified"<<std::endl;
		}
		delete [] dev_in_grad;
	}

	// clean
	operator_destroy(op_pooling);
	gpu_dev->pooling_desc_destroy(pooling_desc);
	gpu_dev->tensor_destroy(t_in);
	gpu_dev->tensor_destroy(t_out);

	operator_destroy(op_pooling_c);
	cpu_dev->pooling_desc_destroy(pooling_desc_c);
	cpu_dev->tensor_destroy(t_in_c);
	cpu_dev->tensor_destroy(t_out_c);
	if(!is_fwd){
		gpu_dev->tensor_destroy(t_in_grad);
		gpu_dev->tensor_destroy(t_out_grad);
		cpu_dev->tensor_destroy(t_in_grad_c);
		cpu_dev->tensor_destroy(t_out_grad_c);
	}
	return 0;
}
#endif

static int conv_driver(int argc, char ** argv){
	arg_parser parser("conv");
	parser.insert_arg(
			"spatial_dim", '_', "2", "convolution spatial dimension (Default-2)", "int");
	parser.insert_arg("forw",
			'F',
			"1",
			"Flag enables fwd, bwd, wrw convolutions"
			"\n0 fwd+bwd+wrw (default)"
			"\n1 fwd only"
			"\n2 bwd only"
			"\n4 wrw only"
			"\n3 fwd+bwd"
			"\n5 fwd+wrw"
			"\n6 bwd+wrw",
			"int");
	parser.insert_arg("batchsize", 'n', "100", "Mini-batch size (Default=100)", "int");
	parser.insert_arg("in_channels", 'c', "3", "Number of Input Channels (Default=3)", "int");
	parser.insert_arg("in_d", '!', "32", "Input Depth (Default=32)", "int");
	parser.insert_arg("in_h", 'H', "32", "Input Height (Default=32)", "int");
	parser.insert_arg("in_w", 'W', "32", "Input Width (Default=32)", "int");
	parser.insert_arg(
			"out_channels", 'k', "32", "Number of Output Channels (Default=32)", "int");
	parser.insert_arg("fil_d", '@', "3", "Filter Depth (Default=3)", "int");
	parser.insert_arg("fil_h", 'y', "3", "Filter Height (Default=3)", "int");
	parser.insert_arg("fil_w", 'x', "3", "Filter Width (Default=3)", "int");
	parser.insert_arg(
			"conv_stride_d", '#', "1", "Convolution Stride for Depth (Default=1)", "int");
	parser.insert_arg(
			"conv_stride_h", 'u', "1", "Convolution Stride for Height (Default=1)", "int");
	parser.insert_arg(
			"conv_stride_w", 'v', "1", "Convolution Stride for Width (Default=1)", "int");
	parser.insert_arg("pad_d", '$', "0", "Zero Padding for Depth (Default=0)", "int");
	parser.insert_arg("pad_h", 'p', "0", "Zero Padding for Height (Default=0)", "int");
	parser.insert_arg("pad_w", 'q', "0", "Zero Padding for Width (Default=0)", "int");
	parser.insert_arg("pad_val", 'r', "0", "Padding Value (Default=0)", "int");
	parser.insert_arg(
			"trans_output_pad_d", '%', "0", "Zero Padding Output for Depth (Default=0)", "int");
	parser.insert_arg(
			"trans_output_pad_h", 'Y', "0", "Zero Padding Output for Height (Default=0)", "int");
	parser.insert_arg(
			"trans_output_pad_w", 'X', "0", "Zero Padding Output for Width (Default=0)", "int");
	parser.insert_arg("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
	parser.insert_arg("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
	parser.insert_arg("verification_cache",
			'C',
			"",
			"Use specified directory to cache verification data. Off by default.",
			"string");
	parser.insert_arg("time", 't', "1", "Time Each Layer (Default=1)", "int");
	parser.insert_arg(
			"wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");
	parser.insert_arg("search", 's', "0", "Search Kernel Config (Default=0)", "int");
	parser.insert_arg("printconv", 'P', "1", "Print Convolution Dimensions (Default=1)", "int");
	parser.insert_arg("dump_output", 'o', "0", "Dumps the output buffers (Default=0)", "int");
	parser.insert_arg("in_data", 'd', "", "Input data filename (Default=)", "string");
	parser.insert_arg("weights", 'e', "", "Input weights filename (Default=)", "string");
	parser.insert_arg("bias", 'b', "", "Use Bias (Default=0)", "int");
	parser.insert_arg(
			"mode", 'm', "cross_correlation",
			"Convolution Mode (conv/cross_correlation, trans) (Default=conv/cross_correlation)", "str");
	parser.insert_arg(
			"pad_mode", 'z', "default", "Padding Mode (same, valid, default) (Default=default)", "str");
	parser.insert_arg("tensor_vect",
			'Z',
			"0",
			"tensor vectorization type (none, vect_c, vect_n) (Default=0)",
			"int");
	parser.insert_arg("dilation_d", '^', "1", "Dilation of Filter Depth (Default=1)", "int");
	parser.insert_arg("dilation_h", 'l', "1", "Dilation of Filter Height (Default=1)", "int");
	parser.insert_arg("dilation_w", 'j', "1", "Dilation of Filter Width (Default=1)", "int");
	parser.insert_arg("in_bias", 'a', "", "Input bias filename (Default=)", "string");
	parser.insert_arg("group_count", 'g', "1", "Number of Groups (Default=1)", "int");
	parser.insert_arg("dout_data",
			'D',
			"",
			"dy data filename for backward weight computation (Default=)",
			"string");
	parser.insert_arg("solution",
			'S',
			"-1",
			"Use immediate mode, run solution with specified id."
			"\nAccepts integer argument N:"
			"\n=0 Immediate mode, build and run fastest solution"
			"\n>0 Immediate mode, build and run solution_id = N"
			"\n<0 Use Find() API (Default=-1)",
			"int");

	// parse arg
	if(!parser.parse(argc, argv)) return -1;
	//parser.dump_parsed();

	//int spatial_dim = ;
	if (parser.get_arg_int("_") != 2)
		assert("Only convolution-2d supported now.");

	// get param from arg
	size_t batch    = (size_t)parser.get_arg_int("n");
	size_t input_c  = (size_t)parser.get_arg_int("c");
	size_t output_c  = parser.get_arg_int("k");
	size_t input_h  = (size_t)parser.get_arg_int("H");
	size_t input_w  = (size_t)parser.get_arg_int("W");
	int pad_h  = parser.get_arg_int("p");
	int pad_w  = parser.get_arg_int("q");
	int stride_h   = parser.get_arg_int("u");
	int stride_w   = parser.get_arg_int("v");
	int fil_w    = parser.get_arg_int("x");
	int fil_h    = parser.get_arg_int("y");
	int groups   = parser.get_arg_int("g");
	int dilation_h = parser.get_arg_int("l");
	int dilation_w = parser.get_arg_int("j");
	int bias = parser.get_arg_int("b");
	std::string cmode = parser.get_arg("m");
	int fmode = parser.get_arg_int("F");
	int is_fwd = (fmode == 0 || fmode == 1 || fmode == 3 || fmode == 5) ? 1 : 0;
	int is_bwd = (fmode == 0 || fmode == 2 || fmode == 3 || fmode == 6) ? 1 : 0;
	int is_wrw = (fmode == 0 || fmode == 4 || fmode == 5 || fmode == 6) ? 1 : 0;
	int time_enabled = parser.get_arg_int("t");
	int num_iterations = parser.get_arg_int("i");

	debug_msg("Conv2d(input=(%lu,%lu,%lu,%lu), output_channels=(%lu), kernel_size=(%d,%d), "
			"stride=(%d,%d), padding=(%d,%d), bias=%s\n\tgroups=(%d), "
			"dilation=(%d,%d))\n",
			batch, input_c, input_h, input_w, output_c, fil_h, fil_w,
			stride_h, stride_w, pad_h, pad_w, bias > 0 ? "True" : "False",
			groups, dilation_h, dilation_w);

	std::string in_x = parser.get_arg("d");
	std::string in_w = parser.get_arg("e");
	std::string in_b = parser.get_arg("a");
	std::string in_dy = parser.get_arg("D");

	int is_save_out = parser.get_arg_int("o");

	assert( (input_c%groups)==0 && "group conv must evenly devide input channel!");
	//    assert( (filters%groups)==0 && "group conv must evenly devide filter number!");

	convolution_mode conv_mode;
	if(cmode=="conv") conv_mode = CONVOLUTION_CONV;
	else if (cmode == "cross_correlation") conv_mode = CONVOLUTION_CROSS_CORRELATION;
	else {std::cout<<"unsupported conv mode "<<cmode<<std::endl; return -1;}

	int _ksize[2] = {fil_h, fil_w};
	int _pad[2] = {pad_h, pad_w};
	int _stride[2] = {stride_h, stride_w};
	int _dilation[2] = {dilation_h, dilation_w};
	convolution_desc_t *conv_desc = gpu_dev->convolution_desc_create(conv_mode, TENSOR_DT_FLOAT,
			_ksize, _stride, _pad, _dilation, 2,
			groups, output_c, input_c, input_h, input_w);

#if 1
	convolution_desc_t * conv_desc_c = cpu_dev->convolution_desc_create(conv_mode, TENSOR_DT_FLOAT,
			_ksize, _stride, _pad, _dilation, 2,
			groups, output_c, input_c, input_h, input_w);
#endif

	tensor_t *t_in, *t_out, *t_filter, *t_in_c, *t_out_c, *t_filter_c;
	tensor_t *t_in_grad, *t_out_grad, *t_filter_grad;
	tensor_t *t_in_grad_c, *t_out_grad_c, *t_filter_grad_c;
	operator_base *op_conv, *op_conv_c;
	size_t t_in_dim[4] = {batch,input_c,input_h,input_w};
	size_t t_filter_dim[4] = {(size_t)output_c, (size_t)(input_c/groups), (size_t)fil_h, (size_t)fil_w};
	size_t t_out_dim[4];

	// create gpu tensors
	op_conv = operator_create(gpu_dev, OP_CONV, conv_desc);

	t_in = gpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_conv->input = t_in;
	t_filter = gpu_dev->tensor_create(t_filter_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_conv->filter = t_filter;
	op_conv->infer_shape(t_out_dim);
	t_out = gpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_conv->output = t_out;

	if(is_bwd){
		t_in_grad = gpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		t_out_grad = gpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		op_conv->input_grad = t_in_grad;
		op_conv->output_grad = t_out_grad;
	}

	if (is_wrw) {
		t_filter_grad = gpu_dev->tensor_create(t_filter_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		op_conv->filter_grad = t_filter_grad;
	}

	// create cpu tensors
	//	if (is_fwd) {
	op_conv_c = operator_create(cpu_dev, OP_CONV, conv_desc_c);
	t_in_c = cpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	t_out_c = cpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	t_filter_c = cpu_dev->tensor_create(t_filter_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_conv_c->filter = t_filter_c;
	op_conv_c->input = t_in_c;
	op_conv_c->output = t_out_c;
	//	}

#if 1
	if (is_bwd) {
		t_in_grad_c = cpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		t_out_grad_c = cpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		t_filter_grad_c = cpu_dev->tensor_create(t_filter_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		op_conv_c->input_grad = t_in_grad_c;
		op_conv_c->output_grad = t_out_grad_c;
		op_conv_c->filter_grad = t_filter_grad_c;
	}
#endif

	op_conv->tune_op();
	op_conv->alloc_mem();
	op_conv_c->tune_op();
	op_conv_c->alloc_mem();

	/*
	int out_n, out_c, out_h, out_w;
#ifdef WITH_CUDNN
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(op_conv->conv_desc,
				(const cudnnTensorDescriptor_t)op_conv->input->desc,
				(op_convolution_cudnn *)op_conv->filter_desc, &out_n, &out_c, &out_h, &out_w));
	t_out_dim[0] = out_n;
	t_out_dim[1] = out_c;
	t_out_dim[2] = out_h;
	t_out_dim[3] = out_w;
	t_out_c = cpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_conv_c->output = t_out_c;
#endif
*/

	if (in_x != "" && !readFromTxt((float*)t_in_c->mem, t_in_c->elem(), in_x.c_str()))
		rand_float((float*)t_in_c->mem, t_in_c->elem());

	if (in_w != "" && !readFromTxt((float*)t_filter_c->mem, t_filter_c->elem(), in_w.c_str()))
		rand_float((float*)t_in_c->mem, t_in_c->elem());

	// host to gpu
	gpu_dev->tensor_copy(t_in, t_in_c->mem, t_in_c->bytes(), TENSOR_COPY_H2D);
	gpu_dev->tensor_copy(t_filter, t_filter_c->mem, t_filter_c->bytes(), TENSOR_COPY_H2D);

	if(is_bwd){
		if (in_dy != "" && !readFromTxt((float*)t_out_grad_c->mem,
					t_out_grad_c->elem(), in_dy.c_str()))
			rand_float((float*)t_out_grad_c->mem, t_out_grad_c->elem());

		gpu_dev->tensor_copy(t_out_grad, t_out_grad_c->mem, t_out_grad_c->bytes(), TENSOR_COPY_H2D);
	}

	device_timer_t * dt = gpu_dev->device_timer_create();

	if (is_fwd) {
		dt->reset();
		dt->start();
		for(int l=0;l<num_iterations;l++){
			op_conv->forward();
		}
		dt->stop();

		if (time_enabled)
			op_conv->print_fwd_time(dt->elapsed() / num_iterations);

		if (is_save_out) {
			float * fwd_out = new float[t_out->elem()];
			gpu_dev->tensor_copy(fwd_out, t_out, t_out->bytes(), TENSOR_COPY_D2H);
			writeToTxt("conv_fwd_out.txt", fwd_out, t_out->elem());
			delete[] fwd_out;
		}
	}

	if (is_bwd) {
		dt->reset();
		dt->start();
		for(int l=0;l<num_iterations;l++){
			op_conv->backward_data();
		}
		dt->stop();

		if (time_enabled)
			op_conv->print_bwd_time(dt->elapsed() / num_iterations);

		if (is_save_out) {
			float * bwd_out = new float[t_in_grad->elem()];
			gpu_dev->tensor_copy(bwd_out, t_in_grad, t_in_grad->bytes(), TENSOR_COPY_D2H);
			writeToTxt("conv_bwd_out.txt", bwd_out, t_in_grad->elem());
			delete[] bwd_out;
		}
	}

#if 0
	if (is_wrw) {
		dt->reset();
		dt->start();
		for(int l=0;l<num_iterations;l++){
			op_conv->backward_filter();
		}
		dt->stop();

		if (time_enabled)
			dynamic_cast<op_convolution*>(op_conv)->print_wrw_time(dt->elapsed() / num_iterations);

		if (is_save_out) {
			float * wrw_out = new float[t_filter_grad->elem()];
			gpu_dev->tensor_copy(wrw_out, t_filter_grad, t_filter_grad->bytes(), TENSOR_COPY_D2H);
			writeToTxt("conv_wrw_out.txt", wrw_out, t_filter_grad->elem());
			delete[] wrw_out;
		}
	}
#endif

#if 0
	for(int l=0;l<LOOP_WARMUP;l++){
		op_conv->forward();
	}
	dt->reset();
	dt->start();
	for(int l=0;l<num_iterations;l++){
		op_conv->forward();
	}
	dt->stop();

	if (time_enabled) {
		std::string fwd_algo_name = dynamic_cast<op_convolution*>(op_conv)->get_fwd_algo_name();
		std::cout << "OpDriver Forward Conv. Algorithm: " << fwd_algo_name << "." << std::endl;
		op_conv->print_fwd_time(dt->elapsed() / num_iterations);
	}

	if (is_save_out) {
		float * dev_out = new float[t_out->elem()];
		gpu_dev->tensor_copy(dev_out, t_out, t_out->bytes(), TENSOR_COPY_D2H);
		dumpBufferToTxt("conv_out.txt", dev_out, t_out->elem());
	}
#endif

#if 0
	if(!is_fwd){
		dt->reset();
		for(int l=0;l<LOOP_WARMUP;l++){
			op_conv->backward_data();
		}
		dt->start();
		for(int l=0;l<LOOP_ITR;l++){
			op_conv->backward_data();
		}
		dt->stop();
		cost_per_loop = dt->elapsed()/LOOP_ITR;
#ifdef EF_PRT
#else
		std::cout<<"convolution bwd_data gpu cost "<<cost_per_loop<<"ms average"<<std::endl;
#endif

		dt->reset();
		for(int l=0;l<LOOP_WARMUP;l++){
			op_conv->backward_filter();
		}
		dt->start();
		for(int l=0;l<LOOP_ITR;l++){
			op_conv->backward_filter();
		}
		dt->stop();
		cost_per_loop = dt->elapsed()/LOOP_ITR;
#ifdef EF_PRT
		std::cout<<"convolution bwd_filter gpu cost "<<cost_per_loop<<"ms average"<<std::endl;
#endif
	}
#endif
	gpu_dev->device_timer_destroy(dt);
	//#define CPU_VALIDATE
	//validation
#ifdef CPU_VALIDATE
	op_conv_c->forward();
	if(!is_fwd)
		op_conv_c->backward();

	// compare
	float * dev_out = new float[t_out->elem()];
	gpu_dev->tensor_copy(dev_out, t_out, t_out->bytes(), TENSOR_COPY_D2H);
	int error_cnt = util_compare_data(dev_out, t_out_c->mem, t_out_c->elem(), TENSOR_DT_FLOAT, 0.001);
	if(error_cnt){
		std::cout<<"convolution fwd compare fail"<<std::endl;
	}else{
		std::cout<<"convolution fwd result verified"<<std::endl;
	}
	delete [] dev_out;
	if(!is_fwd){
		float * dev_in_grad = new float[t_in_grad->elem()];
		gpu_dev->tensor_copy(dev_in_grad, t_in_grad, t_in_grad->bytes(), TENSOR_COPY_D2H);

		//int error_cnt_grad = util_compare_data(dev_in_grad, t_in_grad_c->mem, t_in_grad_c->elem(), TENSOR_DT_FLOAT, 0.001);
		//if(error_cnt_grad){
		//    std::cout<<"convolution bwd compare fail"<<std::endl;
		//}else{
		//    std::cout<<"convolution bwd result verified"<<std::endl;
		//}
		delete [] dev_in_grad;
	}
#endif
	// clean
	operator_destroy(op_conv);
	gpu_dev->convolution_desc_destroy(conv_desc);
	gpu_dev->tensor_destroy(t_in);
	gpu_dev->tensor_destroy(t_out);
	gpu_dev->tensor_destroy(t_filter);

	operator_destroy(op_conv_c);
	cpu_dev->convolution_desc_destroy(conv_desc_c);
	cpu_dev->tensor_destroy(t_in_c);
	cpu_dev->tensor_destroy(t_out_c);
	cpu_dev->tensor_destroy(t_filter_c);
	if(!is_fwd){
		gpu_dev->tensor_destroy(t_in_grad);
		gpu_dev->tensor_destroy(t_out_grad);
		gpu_dev->tensor_destroy(t_filter_grad);
		cpu_dev->tensor_destroy(t_in_grad_c);
		cpu_dev->tensor_destroy(t_out_grad_c);
		cpu_dev->tensor_destroy(t_filter_grad_c);
	}
	return 0;
}

#if 0
static int act_driver(int argc, char ** argv){
	arg_parser parser("act");

	parser.insert_arg("n", "batch", "2");
	parser.insert_arg("c", "channel", "3");
	parser.insert_arg("h", "height", "128");
	parser.insert_arg("w", "width", "128");
	parser.insert_arg("m", "activation mode, "
			"sigmoid relu tanh clipped-relu elu identity", 
			"sigmoid");
	parser.insert_arg("a", "alpha value, used in some activation mode", "1.0");
	parser.insert_arg("f", "forward(1) or backward(0)", "0");

	// parse arg
	if(!parser.parse(argc, argv)) return -1;
	parser.dump_parsed();

	// get param from arg
	size_t n   = (size_t)parser.get_arg_int("n");
	size_t c   = (size_t)parser.get_arg_int("c");
	size_t h   = (size_t)parser.get_arg_int("h");
	size_t w   = (size_t)parser.get_arg_int("w");
	int is_fwd = parser.get_arg_int("f");
	std::string amode = parser.get_arg("m");
	float alpha = parser.get_arg_float("a");

	// start
	activation_mode am = ACTIVATION_SIGMOID;
	if(amode == "sigmoid") am = ACTIVATION_SIGMOID;
	else if(amode == "relu") am = ACTIVATION_RELU;
	else if(amode == "tanh") am = ACTIVATION_TANH;
	else if(amode == "clipped-relu") am = ACTIVATION_CLIPPED_RELU;
	else if(amode == "elu") am = ACTIVATION_ELU;
	else if(amode == "identity") am = ACTIVATION_IDENTITY;
	else {std::cout<<"unsupport activation mode "<<amode<<std::endl; return -1;}

	activation_desc_t * act_desc = gpu_dev->activation_desc_create(am, alpha);
	activation_desc_t * act_desc_c = cpu_dev->activation_desc_create(am, alpha);

	tensor_t *t_in, *t_out, *t_in_c, *t_out_c;
	tensor_t *t_in_grad, *t_out_grad, *t_in_grad_c, *t_out_grad_c;
	operator_base *op_act, *op_act_c;
	size_t t_in_dim[4] = {n,c,h,w};
	size_t t_out_dim[4];

	// create gpu tensors
	op_act = operator_create(gpu_dev, OP_ACTIVATION, act_desc);

	t_in = gpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_act->input = t_in;
	op_act->infer_shape(t_out_dim);
	t_out = gpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_act->output = t_out;
	if(!is_fwd){
		t_in_grad = gpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		t_out_grad = gpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		op_act->input_grad = t_in_grad;
		op_act->output_grad = t_out_grad;
	}

	// create cpu tensors
	op_act_c = operator_create(cpu_dev, OP_ACTIVATION, act_desc_c);
	t_in_c = cpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	t_out_c = cpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
	op_act_c->input = t_in_c;
	op_act_c->output = t_out_c;
	if(!is_fwd){
		t_in_grad_c = cpu_dev->tensor_create(t_in_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		t_out_grad_c = cpu_dev->tensor_create(t_out_dim, 4, TENSOR_DT_FLOAT, TENSOR_LAYOUT_NCHW);
		op_act_c->input_grad = t_in_grad_c;
		op_act_c->output_grad = t_out_grad_c;
	}

	op_act->tune_op();
	op_act->alloc_mem();
	op_act_c->tune_op();
	op_act_c->alloc_mem();

	// prepare input
	rand_float((float*)t_in_c->mem, t_in_c->elem());
	gpu_dev->tensor_copy(t_in, t_in_c->mem, t_in_c->bytes(), TENSOR_COPY_H2D);
	if(!is_fwd){
		rand_float((float*)t_out_grad_c->mem, t_out_grad_c->elem());
		gpu_dev->tensor_copy(t_out_grad, t_out_grad_c->mem, t_out_grad_c->bytes(), TENSOR_COPY_H2D);

		cpu_dev->tensor_set(t_in_grad_c, 0);
		gpu_dev->tensor_set(t_in_grad, 0);
	}

	op_act->forward();
	if(!is_fwd)
		op_act->backward();
	//validation
	op_act_c->forward();
	if(!is_fwd)
		op_act_c->backward();

	// compare
	float * dev_out = new float[t_out->elem()];
	gpu_dev->tensor_copy(dev_out, t_out, t_out->bytes(), TENSOR_COPY_D2H);

	int error_cnt = util_compare_data(dev_out, t_out_c->mem, t_out_c->elem(), TENSOR_DT_FLOAT, 0.001);
	if(error_cnt){
		std::cout<<"activation fwd compare fail"<<std::endl;
	}else{
		std::cout<<"activation fwd result verified"<<std::endl;
	}
	delete [] dev_out;
	if(!is_fwd){
		float * dev_in_grad = new float[t_in_grad->elem()];
		gpu_dev->tensor_copy(dev_in_grad, t_in_grad, t_in_grad->bytes(), TENSOR_COPY_D2H);

		int error_cnt_grad = util_compare_data(dev_in_grad, t_in_grad_c->mem, t_in_grad_c->elem(), TENSOR_DT_FLOAT, 0.001);
		if(error_cnt_grad){
			std::cout<<"activation bwd compare fail"<<std::endl;
		}else{
			std::cout<<"activation bwd result verified"<<std::endl;
		}
		delete [] dev_in_grad;
	}

	// clean
	operator_destroy(op_act);
	gpu_dev->activation_desc_destroy(act_desc);
	gpu_dev->tensor_destroy(t_in);
	gpu_dev->tensor_destroy(t_out);

	operator_destroy(op_act_c);
	cpu_dev->activation_desc_destroy(act_desc_c);
	cpu_dev->tensor_destroy(t_in_c);
	cpu_dev->tensor_destroy(t_out_c);
	if(!is_fwd){
		gpu_dev->tensor_destroy(t_in_grad);
		gpu_dev->tensor_destroy(t_out_grad);
		cpu_dev->tensor_destroy(t_in_grad_c);
		cpu_dev->tensor_destroy(t_out_grad_c);
	}
	return 0;
}
#endif

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
#if 0
	if(op_type == "pooling")
		rtn = pooling_driver(argc, argv);
	else if(op_type == "act")
		rtn = act_driver(argc, argv);
	else if(op_type == "conv")
#else
		if (op_type == "conv")
			rtn = conv_driver(argc, argv);
#endif

	device_destroy(gpu_dev);
	device_destroy(cpu_dev);
	return rtn;
}
