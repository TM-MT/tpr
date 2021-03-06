#pragma once
#include "pm.hpp"

#ifdef __NVCC__
#define CU_CHECK(expr)                                                     \
    {                                                                      \
        cudaError_t t = expr;                                              \
        if (t != cudaSuccess) {                                            \
            fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", \
                    cudaGetErrorString(t), t, __FILE__, __LINE__);         \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    }

namespace pmcpp {

/**
 * @brief      Timer for CUDA Programs
 *
 *             Example:
 * @code{.cpp}
 * time_ms elapsed = 0;
 * pmcpp::DeviceTimer timer;
 * timer.start();
 * // Run kernel
 * timer.stop_and_elapsed(elapsed);
 * pmcpp::perf_time.push_back(elapsed);
 * @endcode
 */
class DeviceTimer {
    cudaEvent_t ev_start, ev_stop;

   public:
    DeviceTimer() {
        CU_CHECK(cudaEventCreate(&ev_start));
        CU_CHECK(cudaEventCreate(&ev_stop));
    }

    ~DeviceTimer() {
        CU_CHECK(cudaEventDestroy(ev_start));
        CU_CHECK(cudaEventDestroy(ev_stop));
    }

    void start() { CU_CHECK(cudaEventRecord(ev_start, cudaEventDefault)); }

    void stop() {
        CU_CHECK(cudaEventRecord(ev_stop, cudaEventDefault));
        cudaEventSynchronize(ev_stop);
    }

    void get_elapsed_time(time_ms &elapsed) {
        cudaEventElapsedTime(&elapsed, ev_start, ev_stop);
    }

    void stop_and_elapsed(time_ms &elapsed) {
        stop();
        get_elapsed_time(elapsed);
    }
};
}  // namespace pmcpp

#undef CU_CHECK
#endif
