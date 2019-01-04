#ifndef __MATH_H
#define __MATH_H



namespace math{

void im2col(const float* data_im,
                        int im_c, int im_h, int im_w,
                        int dilation_h, int dilation_w,
                        int ksize_h, int ksize_w,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        float* data_col);

typedef enum {
    CblasRowMajor,
    CblasColMajor
} CBLAS_LAYOUT;

typedef enum {
    CblasTrans,
    CblasNoTrans,
    CblasConjTrans,
    CblasConjNoTrans,
} CBLAS_TRANSPOSE;

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
    float *c, const int ldc);


}

#endif
