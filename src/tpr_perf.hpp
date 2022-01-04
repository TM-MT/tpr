#pragma once
#include <array>
#include <string>

/**
 * Helper Functions for Performance Monitoring TPR
 */
namespace tprperf {
// Labels
// c++ does not allow to hold string
enum Labels {
    st1 = 0,
    st2,
    st3,
};

static std::array<const char*, 3> section_names = {"st1", "st2", "st3"};
static std::array<std::string, 3> display_labels;

void init(std::string const& prefix);
void start(tprperf::Labels lb);
void stop(tprperf::Labels lb, double fp = 0.0);
}  // namespace tprperf
