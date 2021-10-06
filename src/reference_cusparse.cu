#include "reference_cusparse.cuh"

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