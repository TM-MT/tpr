#include <array>
#include "pm.hpp"
#include "PerfMonitor.h"
#include "tpr_perf.hpp"

/**
 * Helper Functions for Performance Monitoring TPR
 */
namespace tprperf {
    /**
     * @brief Init Function for Perf.
     *
     * @param n Size of Equation
     * @param s
     */
    void init(int n, int s) {
        #ifdef TPR_PERF
        // Initialize PerfMonitor and set labels
        for (unsigned long int i = 0; i < section_names.size(); i++) {
            auto format = std::string("TPR_n_");
            auto gen_label = format.replace(4, 1, std::to_string(s))
                                .append(section_names[i]);
            display_labels[i] = gen_label;
            pmcpp::pm.setProperties(gen_label, pmcpp::pm.CALC);
        }
        #endif
    }

    /**
     * @brief Helper function of pm.start()
     *
     * @param lb Label
     */
    void start(tprperf::Labels lb) {
        #ifdef TPR_PERF
        return pmcpp::pm.start(display_labels[static_cast<int>(lb)]);
        #endif
    }

    /**
     * @brief Helper function of pm.stop()
     *
     * @param lb Label
     * @param fp user provided count
     */
    void stop(tprperf::Labels lb, double fp) {
        #ifdef TPR_PERF
        return pmcpp::pm.stop(display_labels[static_cast<int>(lb)], fp);
        #endif
    }
}
