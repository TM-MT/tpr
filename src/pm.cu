#include <stdio.h>
#include <stdlib.h>

#include <initializer_list>
#include <string>

#include "effolkronium/random.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "pm.cuh"
#include "ptpr.cuh"
#include "reference_cusparse.cuh"
#include "system.hpp"
#include "tpr.cuh"

// std::mt19937 base pseudo-random
using Random = effolkronium::random_static;

namespace pmcpp {
Perf perf_time;
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

Solver str2Solver(std::string solver) {
    to_lower(solver);
    if (solver.compare(std::string("pcr")) == 0) {
        return Solver::PCR;
    } else if (solver.compare(std::string("tpr")) == 0) {
        return Solver::TPR;
    } else if (solver.compare(std::string("ptpr")) == 0) {
        return Solver::PTPR;
    } else if (solver.compare(std::string("cusparse")) == 0) {
        return Solver::cuSparse;
    } else {
        std::cerr << "Solver Not Found.\n";
        abort();
    }
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

    trisys::ExampleFixedInput input(n);

    // 1. setup the system by calling assign()
    // 2. set the system
    // 3. measure
    switch (solver) {
        case pmcpp::Solver::TPR: {
            for (int i = 0; i < iter_times; i++) {
                input.assign();
                TPR_CU::tpr_cu(input.sys.a, input.sys.c, input.sys.rhs,
                               input.sys.diag, n, s);
            }
        } break;
        case pmcpp::Solver::PTPR: {
            for (int i = 0; i < iter_times; i++) {
                input.assign();
                PTPR_CU::ptpr_cu(input.sys.a, input.sys.c, input.sys.rhs,
                                 input.sys.diag, n, s);
            }
        } break;
        case pmcpp::Solver::PCR: {
            for (int i = 0; i < iter_times; i++) {
                input.assign();
                PTPR_CU::pcr_cu(input.sys.a, input.sys.c, input.sys.rhs,
                                input.sys.diag, n);
            }
        } break;
        case pmcpp::Solver::cuSparse: {
            REFERENCE_CUSPARSE rfs(n);
            for (int i = 0; i < iter_times; i++) {
                input.assign();
                rfs.solve(input.sys.a, input.sys.c, input.sys.rhs,
                          input.sys.diag, n);
            }
        } break;
    }

    pmcpp::perf_time.display();
}
