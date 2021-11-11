#pragma once
#ifdef __NVCC__
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

class REFERENCE_CUSPARSE {
    float *diag;
    cusparseHandle_t cusparseH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    // device memory
    float *dl, *d, *du, *db;
    void *pBuffer;
    int size_of_mem;

   public:
    REFERENCE_CUSPARSE(int n);
    ~REFERENCE_CUSPARSE();
    void solve(float *a, float *c, float *rhs, float *x, int n);
};
#else
class REFERENCE_CUSPARSE {
   public:
    REFERENCE_CUSPARSE(int n);
    ~REFERENCE_CUSPARSE();
    void solve(float *a, float *c, float *rhs, float *x, int n);
};
#endif
