/**
 * @brief      Benchmark program for TPR etc.
 *
 * @author     TM-MT
 * @date       2021
 */
#include "pm.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <initializer_list>
#include <string>

#include "PerfMonitor.h"
#include "lib.hpp"
#include "main.hpp"
#include "pcr.hpp"
#include "ptpr.hpp"
#include "system.hpp"
#include "tpr.hpp"

#ifdef BUILD_CUDA
#include "pm.cuh"
#include "ptpr.cuh"
#include "reference_cusparse.cuh"
#include "system.hpp"
#include "tpr.cuh"
#endif

namespace pmcpp {
Perf perf_time;
pm_lib::PerfMonitor pm = pm_lib::PerfMonitor();

/**
 * @brief Command Line Args
 * @details Define Command Line Arguments
 */
struct Options {
    // size of system
    int n;
    // size of slice
    int s;
    // Iteration Times
    int iter;
    // Solver
    Solver solver;
};

/**
 * @brief      Convert string representation to pmcpp::Solver
 *
 * @param[in]  solver  The solver name in string
 *
 * @return     pmcpp::Solver or abort if not found
 */
Solver str2Solver(std::string solver) {
    to_lower(solver);
    if (solver.compare(std::string("pcr")) == 0) {
        return Solver::PCR;
    } else if (solver.compare(std::string("tpr")) == 0) {
        return Solver::TPR;
    } else if (solver.compare(std::string("ptpr")) == 0) {
        return Solver::PTPR;
    }
#ifdef BUILD_CUDA
    // only available options on `BUILD_CUDA`
    else if (solver.compare(std::string("cutpr")) == 0) {
        return Solver::CUTPR;
    } else if (solver.compare(std::string("cuptpr")) == 0) {
        return Solver::CUPTPR;
    } else if (solver.compare(std::string("cusparse")) == 0) {
        return Solver::CUSPARSE;
    }
#endif
    else {
        std::cerr << "Solver Not Found.\n";
        abort();
    }
}

/**
 * @brief      Use PMlib or not.
 *
 * @param      solver  The solver
 *
 * @return     True if using PMlib
 */
bool use_pmlib(Solver &solver) {
    // TPR, PCR, PTPR takes 0, 1, 2 respectively
    return static_cast<int>(solver) < 3;
}

void to_lower(std::string &s1) {
    transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
}

/**
 * @brief parse command line args
 *
 * @param argc [description]
 * @param argv [description]
 */
Options parse(int argc, char *argv[]) {
    // assert args are given in Options order
    if (argc != 5) {
        std::cerr << "Invalid command line args.\n";
        std::cerr << "Usage: " << argv[0] << " n s iter solver\n";
        exit(EXIT_FAILURE);
    }

    Options ret;
    ret.n = atoi(argv[1]);
    ret.s = atoi(argv[2]);
    ret.iter = atoi(argv[3]);
    ret.solver = pmcpp::str2Solver(std::string(argv[4]));

    return ret;
}
}  // namespace pmcpp

int main(int argc, char *argv[]) {
    int n, s, iter_times;
    pmcpp::Solver solver;
    // Parse Command Line Args
    {
        pmcpp::Options in = pmcpp::parse(argc, argv);
        n = in.n;
        s = in.s;
        iter_times = in.iter;
        solver = in.solver;
    }

    trisys::ExampleRandomRHSInput input(n);

    pmcpp::pm.initialize(100);

    // 1. setup the system by calling assign()
    // 2. set the system
    // 3. measure
    switch (solver) {
        case pmcpp::Solver::TPR: {
            auto tpr_all_label = std::string("TPR_").append(std::to_string(s));
            pmcpp::pm.setProperties(tpr_all_label, pmcpp::pm.CALC);

            TPR t(input.sys.n, s);
            for (int i = 0; i < iter_times; i++) {
                input.assign();
#pragma acc data copy(                                                  \
    input.sys.a[:n], input.sys.diag[:n], input.sys.c[:n], input.sys.rhs \
    [:n], input.sys.n)
                {
                    t.set_tridiagonal_system(input.sys.a, input.sys.c,
                                             input.sys.rhs);
                    pmcpp::pm.start(tpr_all_label);
                    int flop_count = t.solve();
                    flop_count += t.get_ans(input.sys.diag);
                    pmcpp::pm.stop(tpr_all_label, flop_count);
                }
            }
        } break;
        case pmcpp::Solver::PTPR: {
            auto tpr_all_label = std::string("PTPR_").append(std::to_string(s));
            pmcpp::pm.setProperties(tpr_all_label, pmcpp::pm.CALC);

            PTPR t(input.sys.n, s);
            for (int i = 0; i < iter_times; i++) {
                input.assign();
#pragma acc data copy(                                                  \
    input.sys.a[:n], input.sys.diag[:n], input.sys.c[:n], input.sys.rhs \
    [:n], input.sys.n)
                {
                    t.set_tridiagonal_system(input.sys.a, input.sys.c,
                                             input.sys.rhs);
                    pmcpp::pm.start(tpr_all_label);
                    int flop_count = t.solve();
                    flop_count += t.get_ans(input.sys.diag);
                    pmcpp::pm.stop(tpr_all_label, flop_count);
                }
            }
        } break;
        case pmcpp::Solver::PCR: {
            auto pcr_label = std::string("PCR");
            pmcpp::pm.setProperties(pcr_label);
            for (int i = 0; i < iter_times; i++) {
                input.assign();

#pragma acc data copy(                                                  \
    input.sys.a[:n], input.sys.diag[:n], input.sys.c[:n], input.sys.rhs \
    [:n], input.sys.n)
                {
                    PCR p(input.sys.a, input.sys.diag, input.sys.c,
                          input.sys.rhs, input.sys.n);
                    pmcpp::pm.start(pcr_label);
                    int flop_count = p.solve();
                    flop_count += p.get_ans(input.sys.diag);
                    pmcpp::pm.stop(pcr_label, flop_count);
                }
            }
        } break;
#ifdef BUILD_CUDA
        case pmcpp::Solver::CUTPR: {
            for (int i = 0; i < iter_times; i++) {
                input.assign();
                TPR_CU::tpr_cu(input.sys.a, input.sys.c, input.sys.rhs,
                               input.sys.diag, n, s);
            }
        } break;
        case pmcpp::Solver::CUPTPR: {
            for (int i = 0; i < iter_times; i++) {
                input.assign();
                PTPR_CU::ptpr_cu(input.sys.a, input.sys.c, input.sys.rhs,
                                 input.sys.diag, n, s);
            }
        } break;
        case pmcpp::Solver::CUSPARSE: {
            REFERENCE_CUSPARSE rfs(n);
            for (int i = 0; i < iter_times; i++) {
                input.assign();
                rfs.solve(input.sys.a, input.sys.c, input.sys.rhs,
                          input.sys.diag, n);
            }
        } break;
#endif
    }

    // CPU programs are measured by pmlib
    if (pmcpp::use_pmlib(solver)) {
        pmcpp::pm.print(stdout, std::string(""), std::string(), 1);
        pmcpp::pm.printDetail(stdout, 0, 1);
    } else {
        pmcpp::perf_time.display();
    }
}
