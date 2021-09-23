#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <string>
#include <algorithm>

#include "PerfMonitor.h"
#include "lib.hpp"
#include "pcr.hpp"
#include "ptpr.hpp"


namespace pmcpp {
    enum class Solver {
        TPR,
        PCR,
        PTPR,
    };
    extern pm_lib::PerfMonitor pm;

    void to_lower(std::string &s1);
    Solver str2Solver(std::string &solver);
}
