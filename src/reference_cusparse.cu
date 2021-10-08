#include "reference_cusparse.cuh"

#ifdef TPR_PERF
#include "pm.cuh"
#endif

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

REFERENCE_CUSPARSE::REFERENCE_CUSPARSE(int n) {
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

REFERENCE_CUSPARSE::~REFERENCE_CUSPARSE() {
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

/**
 * @brief      Helper function for cusparse::cusparseSgtsv2_nopivot (CR+PCR)
 *
 * @param      a     { parameter_description }
 * @param      c     { parameter_description }
 * @param      rhs   The right hand side
 * @param      x     { parameter_description }
 * @param[in]  n     { parameter_description }
 */
void REFERENCE_CUSPARSE::solve(float *a, float *c, float *rhs, float *x,
                               int n) {
    /* step 3: prepare data in device */
    CU_CHECK(cudaMemcpy(dl, a, size_of_mem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d, diag, size_of_mem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(du, c, size_of_mem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(db, rhs, size_of_mem, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    // calculate the size of the buffer used in gtsv2_nopivot
    size_t pbuffsize;
    CUSP_CHECK(cusparseSgtsv2_nopivot_bufferSizeExt(cusparseH, n, 1, dl, d, du,
                                                    db, n, &pbuffsize));

    CU_CHECK(cudaMalloc((void **)&pBuffer, pbuffsize));

    cudaDeviceSynchronize();

#ifdef TPR_PERF
    {
        time_ms elapsed = 0;
        pmcpp::DeviceTimer timer;
        timer.start();
#endif
        // execute
        CUSP_CHECK(
            cusparseSgtsv2_nopivot(cusparseH, n, 1, dl, d, du, db, n, pBuffer));

#ifdef TPR_PERF
        timer.stop_and_elapsed(elapsed);  // cudaDeviceSynchronize called
        pmcpp::perf_time.push_back(elapsed);
    }
#else
    cudaDeviceSynchronize();
#endif

    CU_CHECK(cudaMemcpy(x, db, size_of_mem, cudaMemcpyDeviceToHost));
    CU_CHECK(cudaFree(pBuffer));

    return;
}
