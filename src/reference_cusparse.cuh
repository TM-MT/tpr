#pragma once
#include "lib.hpp"

#ifdef __NVCC__
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

class REFERENCE_CUSPARSE {
    cusparseHandle_t cusparseH = NULL;
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;
    // device memory
    real *dl, *d, *du, *db;
    void* pBuffer;
    int size_of_mem;

   public:
    REFERENCE_CUSPARSE(int n);
    ~REFERENCE_CUSPARSE();
    void calc_buffer_size(cusparseHandle_t handle, int m, int n, const real* dl,
                          const real* d, const real* du, const real* B, int ldb,
                          size_t* bufferSizeInBytes);
    void gtsv(cusparseHandle_t handle, int m, int n, const real* dl,
              const real* d, const real* du, real* B, int ldb, void* pBuffer);
    void solve(real* a, real* diag, real* c, real* rhs, real* x, int n);
    void solve(real* a, real* c, real* rhs, real* x, int n);
};
#else
class REFERENCE_CUSPARSE {
   public:
    REFERENCE_CUSPARSE(int n);
    ~REFERENCE_CUSPARSE();
    void solve(real* a, real* diag, real* c, real* rhs, real* x, int n);
    void solve(real* a, real* c, real* rhs, real* x, int n);
};
#endif
