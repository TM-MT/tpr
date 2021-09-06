#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <string>
#include <algorithm>

#include "PerfMonitor.h"
#include "lib.hpp"
#include "pcr.hpp"
#include "tpr.hpp"


namespace pmcpp {
    enum class Solver {
        TPR,
        PCR,

    };
    void to_lower(std::string &s1);
    Solver str2Solver(std::string &solver);
}
