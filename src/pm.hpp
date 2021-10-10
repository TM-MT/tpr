#pragma once
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "PerfMonitor.h"
#include "lib.hpp"
#include "pcr.hpp"
#include "ptpr.hpp"

namespace pmcpp {
enum class Solver {
    TPR,
    PCR,
    PTPR,
#ifdef BUILD_CUDA
    CUTPR,
    CUPTPR,
    CUSPARSE,
#endif
};
extern pm_lib::PerfMonitor pm;

void to_lower(std::string &s1);
Solver str2Solver(std::string &solver);

using time_ms = float;
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

class DeviceTimer;
}  // namespace pmcpp
