#pragma once
#include <vector>
using time_ms = float;

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

enum class Solver {
    TPR,
    PCR,
    PTPR,
};

void to_lower(std::string &s1);
Solver str2Solver(std::string &solver);
class Perf {
    std::vector<time_ms> perf_time;

   public:
    void display() {
        std::cout << "sum    [ms]: " << sum() << "\n";
        std::cout << "average[ms]: " << average() << "\n";
        std::cout << "variance   : " << variance() << "\n";
    }

    void display_all(std::string sep = "ms, ") {
        for (std::vector<time_ms>::iterator it = this->perf_time.begin();
             it != this->perf_time.end(); it++) {
            std::cout << *it << sep;
        }
        std::cout << "\n";
    }

    time_ms sum() { return static_cast<float>(sum_d()); }

    time_ms average() { return static_cast<float>(average_d()); }

    time_ms variance() {
        double ave = average_d();
        double tmp = 0.0;
        for (auto &t : this->perf_time) {
            tmp += (t - ave) * (t - ave);
        }
        double n = static_cast<double>(perf_time.size());
        double ret = tmp / n;
        return ret;
    }

    void push_back(time_ms t) { this->perf_time.push_back(t); }

   private:
    double sum_d() {
        double tmp = 0.0;
        for (auto &t : this->perf_time) {
            tmp += t;
        }
        return tmp;
    }

    double average_d() {
        double s = sum_d();
        double len = static_cast<double>(perf_time.size());
        return s / len;
    }
};

extern Perf perf_time;

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
