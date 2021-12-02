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

/**
 * @brief      Constructor of REFERENCE_CUSPARSE
 *
 *             Preparation for cuSPARSE routine call
 * 1. create cuspase/cublas handle and bind a stream
 * 2. allocate device memory
 *
 * @param[in]  n     The order of A
 */
REFERENCE_CUSPARSE::REFERENCE_CUSPARSE(int n) {
    diag = (real*)malloc(n * sizeof(real));
    for (int i = 0; i < n; i++) {
        diag[i] = 1.0;
    }

    size_of_mem = n * sizeof(real);

    /* step 1: create cusparse/cublas handle, bind a stream */
    CU_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamDefault));

    CUSP_CHECK(cusparseCreate(&cusparseH));

    CUSP_CHECK(cusparseSetStream(cusparseH, stream));
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: allocate device memory */
    CU_CHECK(cudaMalloc((void**)&dl, size_of_mem));
    CU_CHECK(cudaMalloc((void**)&d, size_of_mem));
    CU_CHECK(cudaMalloc((void**)&du, size_of_mem));
    CU_CHECK(cudaMalloc((void**)&db, size_of_mem));
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
 * @brief
 * https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_nopivot_bufferSize
 *
 * @param[in]  handle             The handle
 * @param[in]  m                  { parameter_description }
 * @param[in]  n                  { parameter_description }
 * @param[in]  dl                 { parameter_description }
 * @param[in]  d                  { parameter_description }
 * @param[in]  du                 { parameter_description }
 * @param[in]  B                  { parameter_description }
 * @param[in]  ldb                The ldb
 * @param      bufferSizeInBytes  The buffer size in bytes
 */
void REFERENCE_CUSPARSE::calc_buffer_size(cusparseHandle_t handle, int m, int n,
                                          const real* dl, const real* d,
                                          const real* du, const real* B,
                                          int ldb, size_t* bufferSizeInBytes) {
#ifdef _REAL_IS_DOUBLE_
#define BUF_CALC_FUNC cusparseDgtsv2_nopivot_bufferSizeExt
#else
#define BUF_CALC_FUNC cusparseSgtsv2_nopivot_bufferSizeExt
#endif
    CUSP_CHECK(
        BUF_CALC_FUNC(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes));
#undef BUF_CALC_FUNC
}

/**
 * @brief      https://docs.nvidia.com/cuda/cusparse/index.html#gtsv2_nopivot
 *
 * @param[in]  handle   The handle
 * @param[in]  m        { parameter_description }
 * @param[in]  n        { parameter_description }
 * @param[in]  dl       { parameter_description }
 * @param[in]  d        { parameter_description }
 * @param[in]  du       { parameter_description }
 * @param      B        { parameter_description }
 * @param[in]  ldb      The ldb
 * @param      pBuffer  The buffer
 */
void REFERENCE_CUSPARSE::gtsv(cusparseHandle_t handle, int m, int n,
                              const real* dl, const real* d, const real* du,
                              real* B, int ldb, void* pBuffer) {
#ifdef _REAL_IS_DOUBLE_
#define GTSV cusparseDgtsv2_nopivot
#else
#define GTSV cusparseSgtsv2_nopivot
#endif

    // execute
    CUSP_CHECK(GTSV(handle, m, n, dl, d, du, B, ldb, pBuffer));

#undef GTSV
}

/**
 * @brief      Helper function for cusparse::cusparseSgtsv2_nopivot (CR+PCR)
 *
 *             Solve A*x = B by `cusparseSgtsv2_nopivot`, where A is an n-by-n
 *             tridiagonal matrix, B is the right-hand-side vector of legth n
 *
 * @note       Only works in a block.
 *
 * @param[in]  a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                   0.0
 * @param[in]  rhs   rhs[0:n] The right-hand-side of the equation.
 * @param[out] x     x[0:n] for the solution
 * @param[in]  n     The order of A. `n` should be power of 2
 */
void REFERENCE_CUSPARSE::solve(real* a, real* c, real* rhs, real* x, int n) {
    /* step 3: prepare data in device */
    CU_CHECK(cudaMemcpy(dl, a, size_of_mem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(d, diag, size_of_mem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(du, c, size_of_mem, cudaMemcpyHostToDevice));
    CU_CHECK(cudaMemcpy(db, rhs, size_of_mem, cudaMemcpyHostToDevice));

    cudaDeviceSynchronize();

    // calculate the size of the buffer used in gtsv2_nopivot
    size_t pbuffsize;
    calc_buffer_size(cusparseH, n, 1, dl, d, du, db, n, &pbuffsize);

    CU_CHECK(cudaMalloc((void**)&pBuffer, pbuffsize));

    cudaDeviceSynchronize();

#ifdef TPR_PERF
    {
        time_ms elapsed = 0;
        pmcpp::DeviceTimer timer;
        timer.start();
#endif

        gtsv(cusparseH, n, 1, dl, d, du, db, n, pBuffer);

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
