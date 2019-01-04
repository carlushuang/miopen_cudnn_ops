#include "math.hpp"
#include <iostream>
namespace math {


/*
* input image is c*h*w, filter is -> (num_filter)*(c*ksize*ksize)
*
*   c*ksize*ksize                        out_h*out_w
*  +-----------+                         +---------+
*  |           |                         |         |
*  |  1 filter | num_filter   <-dot->    |         |
*  |           |                         | input   | c*ksize*ksize
*  +-----------+                         |         |
*                                        |         |
*                                        +---------+
*
*  im2col need get image as (c*ksize*ksize) * (out_h*out_w)
*/

void im2col(const float* data_im,
                int im_c, int im_h, int im_w,
                int dilation_h, int dilation_w,
                int ksize_h, int ksize_w,
                int stride_h, int stride_w,
                int pad_h, int pad_w,
                float* data_col)
{
    // https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    auto conv_size_func = [](int in_size, int pad, int dilation, int ksize, int stride){
        return (in_size + 2*pad- dilation*(ksize-1) -1)/stride + 1;
    };
    int out_h = conv_size_func(im_h, pad_h, dilation_h, ksize_h, stride_h);
    int out_w = conv_size_func(im_w, pad_w, dilation_w, ksize_w, stride_w);
    int out_c = im_c * ksize_h * ksize_w;
    for(int c=0;c<out_c;c++){
        int w_offset = c % ksize_w;
        int h_offset = (c / ksize_w) % ksize_h;
        int im_c_cur = c / (ksize_w*ksize_h);
        for(int h=0; h<out_h;h++){
            int im_h_cur = h * stride_h - pad_h + h_offset * dilation_h;
            for(int w=0;w<out_w;w++){
                int im_w_cur = w * stride_w - pad_w + w_offset * dilation_w;
                int im_idx = im_c_cur*im_h*im_w + im_h_cur*im_w + im_w_cur;
                int out_idx = c*out_h*out_w + h*out_w + w;
                data_col[out_idx] = (im_h_cur < 0 || im_h_cur >= im_h || im_w_cur<0 || im_w_cur >= im_w) ?
                                    (float).0f:data_im[im_idx];
            }
        }
    }
} 


// https://software.intel.com/en-us/mkl-developer-reference-c-cblas-gemm
/*
* a: m row * k col
* b: k row * n col
* c: m row * n col
*/
void cblas_sgemm (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
    const int m, const int n, const int k,
    const float alpha,
    const float *a, const int lda,
    const float *b, const int ldb,
    const float beta,
    float *c, const int ldc)
{
    if(Layout == CblasRowMajor){
        int im, in, ik;
        auto a_idx_func = (transa == CblasNoTrans || transa == CblasConjNoTrans)?
                [](int m_, int k_, int lda_){return m_*lda_+k_;}:
                [](int m_, int k_, int lda_){return k_*lda_+m_;};
        auto b_idx_func = (transb == CblasNoTrans || transb == CblasConjNoTrans)?
                [](int n_, int k_, int ldb_){return k_*ldb_+n_;}:
                [](int n_, int k_, int ldb_){return n_*ldb_+k_;};
        for(im=0;im<m;im++){
            for(in=0;in<n;in++){
                int c_idx = im*ldc+in;
                float c_val = c[c_idx] * beta;
                for(ik=0;ik<k;ik++){
                    int a_idx = a_idx_func(im, ik, lda);
                    int b_idx = b_idx_func(in, ik, ldb);
                    c_val += alpha * a[a_idx] * b[b_idx];
                }
                c[c_idx] = c_val;
            }
        }
    }else{
        int im, in, ik;
        auto a_idx_func = (transa == CblasNoTrans || transa == CblasConjNoTrans)?
                [](int m_, int k_, int lda_){return k_*lda_+m_;}:
                [](int m_, int k_, int lda_){return m_*lda_+k_;};
        auto b_idx_func = (transb == CblasNoTrans || transb == CblasConjNoTrans)?
                [](int n_, int k_, int ldb_){return n_*ldb_+k_;}:
                [](int n_, int k_, int ldb_){return k_*ldb_+n_;};
        for( in=0;in<n;in++ ){
            for( im=0;im<m;im++ ){
                int c_idx = in*ldc+im;
                float c_val = c[c_idx] * beta;
                for( ik=0;ik<k;ik++ ) {
                    int a_idx = a_idx_func(im, ik, lda);
                    int b_idx = b_idx_func(in, ik, ldb);
                    c_val += alpha * a[a_idx] * b[b_idx];
                }
                c[c_idx] = c_val;
            }
        }
    }
}


}
