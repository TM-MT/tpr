#pragma once
#include <vector>
using time_ms = float;

namespace pmcpp {
extern std::vector<time_ms> perf_time;

enum class Solver {
    TPR,
    PCR,
    PTPR,
};

void to_lower(std::string &s1);
Solver str2Solver(std::string &solver);
}
