/**
 * @brief      Example main function
 *
 * @author     TM-MT
 * @date       2021
 */
#include "main.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <initializer_list>
#include <iostream>

#include "PerfMonitor.h"
#include "cr.hpp"
#include "lib.hpp"
#include "pcr.hpp"
#include "pm.hpp"
#include "ptpr.hpp"
#include "system.hpp"
#include "tpr.hpp"

#ifdef BUILD_CUDA
#include "ptpr.cuh"
#include "tpr.cuh"
#endif

pm_lib::PerfMonitor pmcpp::pm = pm_lib::PerfMonitor();

int main() {
    int n = 1024;

    trisys::ExampleFixedInput input(n);

    input.assign();
#pragma acc data copy( \
    input.sys.a[:n], input.sys.c[:n], input.sys.rhs[:n], input.sys.n)
    {
        CR cr(input.sys.a, input.sys.diag, input.sys.c, input.sys.rhs,
              input.sys.n);
        cr.solve();
        cr.get_ans(input.sys.diag);
    }
    print_array(input.sys.diag, n);
    printf("\n");

    input.assign();
#pragma acc data copy( \
    input.sys.a[:n], input.sys.c[:n], input.sys.rhs[:n], input.sys.n)
    {
        PCR pcr(input.sys.a, input.sys.diag, input.sys.c, input.sys.rhs,
                input.sys.n);
        pcr.solve();
        pcr.get_ans(input.sys.diag);
    }
    print_array(input.sys.diag, n);
    printf("\n");

    pmcpp::pm.initialize(100);
    auto tpr_label = std::string("TPR");
    pmcpp::pm.setProperties(tpr_label, pmcpp::pm.CALC);

    for (int s = 4; s <= n; s *= 2) {
        std::cerr << "TPR s=" << s << "\n";
        input.assign();
#pragma acc data copy(                                                  \
    input.sys.a[:n], input.sys.diag[:n], input.sys.c[:n], input.sys.rhs \
    [:n], input.sys.n)
        {
            TPR t(input.sys.a, input.sys.diag, input.sys.c, input.sys.rhs,
                  input.sys.n, s);
            pmcpp::pm.start(tpr_label);
            int flop_count = t.solve();
            flop_count += t.get_ans(input.sys.diag);
            pmcpp::pm.stop(tpr_label, flop_count);
        }
        print_array(input.sys.diag, n);
        printf("\n");
    }

    tpr_label = std::string("PTPR");
    pmcpp::pm.setProperties(tpr_label, pmcpp::pm.CALC);

    for (int s = 4; s <= n; s *= 2) {
        std::cerr << "PTPR s=" << s << "\n";
        input.assign();
#pragma acc data copy(                                                  \
    input.sys.a[:n], input.sys.diag[:n], input.sys.c[:n], input.sys.rhs \
    [:n], input.sys.n)
        {
            PTPR t(input.sys.a, input.sys.diag, input.sys.c, input.sys.rhs,
                   input.sys.n, s);
            pmcpp::pm.start(tpr_label);
            int flop_count = t.solve();
            flop_count += t.get_ans(input.sys.diag);
            pmcpp::pm.stop(tpr_label, flop_count);
        }
        print_array(input.sys.diag, n);
        printf("\n");
    }

#ifdef BUILD_CUDA
    {
        // CUDA program examples
        TPR_ANS ans1(n), ans2(n);
        for (int s = 128; s <= std::min(n, 1024); s *= 2) {
            input.assign();
            ans1.s = s;
            TPR_CU::tpr_cu(input.sys.a, input.sys.c, input.sys.rhs, ans1.x, n,
                           s);

            // check the answer
            if (s > 128 && ans1 != ans2) {
                std::cout << "TPR(" << ans1.n << "," << ans1.s << ") and TPR("
                          << ans2.n << "," << ans2.s
                          << ") has different answer.\n";
                std::cout << "TPR(" << ans1.n << "," << ans1.s << ")\n";
                ans1.display(std::cout);
                std::cout << "TPR(" << ans2.n << "," << ans2.s << ")\n";
                ans2.display(std::cout);
            }
            ans2 = ans1;
        }

        for (int s = 128; s <= std::min(n, 1024); s *= 2) {
            input.assign();
            ans1.s = s;
            PTPR_CU::ptpr_cu(input.sys.a, input.sys.c, input.sys.rhs, ans1.x, n,
                             s);

            // check the answer
            if (s > 128 && ans1 != ans2) {
                std::cout << "PTPR(" << ans1.n << "," << ans1.s << ") and PTPR("
                          << ans2.n << "," << ans2.s
                          << ") has different answer.\n";
                std::cout << "PTPR(" << ans1.n << "," << ans1.s << ")\n";
                ans1.display(std::cout);
                std::cout << "PTPR(" << ans2.n << "," << ans2.s << ")\n";
                ans2.display(std::cout);
            }
            ans2 = ans1;
        }
    }
#endif

    pmcpp::pm.print(stderr, std::string(""), std::string(), 1);
    pmcpp::pm.printDetail(stderr, 0, 1);
}
