#pragma once
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <iostream>

#include "pm.cuh"

/**
 * @brief      check `cudaError_t`
 *
 * @param      expr  The expression
 */
#define CU_CHECK(expr)                                                     \
    {                                                                      \
        cudaError_t t = expr;                                              \
        if (t != cudaSuccess) {                                            \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", \
                    cudaGetErrorString(t), t, __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

/**
 * @brief      check `cusparseStatus_t`
 *
 * @param      expr  The expression
 */
#define CUSP_CHECK(expr)                                                     \
    {                                                                        \
        cusparseStatus_t t = expr;                                           \
        if (t != CUSPARSE_STATUS_SUCCESS) {                                  \
            fprintf(stderr,                                                  \
                    "[CUSPARSE][Error] %s (error code: %d) at %s line %d\n", \
                    cusparseGetErrorString(t), t, __FILE__, __LINE__);       \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    }

/**
 * @brief      check `cublasStatus_t`
 *
 * @param      expr  The expression
 */
#define CUBLAS_CHECK(expr)                                                 \
    {                                                                      \
        cublasStatus_t t = expr;                                           \
        if (t != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr,                                                \
                    "[CUBLAS][Error] (error code: %d) at %s line %d\n", t, \
                    __FILE__, __LINE__);                                   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

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
    REFERENCE_CUSPARSE(int n) {
        diag = (float *)malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) {
            diag[i] = 1.0;
        }

        size_of_mem = n * sizeof(float);

        /* step 1: create cusparse/cublas handle, bind a stream */
        CU_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

        CUSP_CHECK(cusparseCreate(&cusparseH));

        CUSP_CHECK(cusparseSetStream(cusparseH, stream));
        CUBLAS_CHECK(cublasCreate(&cublasH));
        CUBLAS_CHECK(cublasSetStream(cublasH, stream));

        /* step 2: allocate device memory */
        CU_CHECK(cudaMalloc((void **)&dl, size_of_mem));
        CU_CHECK(cudaMalloc((void **)&d, size_of_mem));
        CU_CHECK(cudaMalloc((void **)&du, size_of_mem));
        CU_CHECK(cudaMalloc((void **)&db, size_of_mem));
    }

    ~REFERENCE_CUSPARSE() {
        // free
        CU_CHECK(cudaFree(dl));
        CU_CHECK(cudaFree(d));
        CU_CHECK(cudaFree(du));
        CU_CHECK(cudaFree(db));
        if (cusparseH) cusparseDestroy(cusparseH);
        if (cublasH) cublasDestroy(cublasH);
        if (stream) cudaStreamDestroy(stream);

        cudaDeviceReset();

        free(diag);
    }

    void solve(float *a, float *c, float *rhs, float *x, int n);
};
