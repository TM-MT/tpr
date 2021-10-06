#include <stdio.h>
#include <stdlib.h>

#include <initializer_list>
#include <string>

#include "effolkronium/random.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "ptpr.cuh"
#include "reference_cusparse.cuh"
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

    struct TRIDIAG_SYSTEM *sys =
        (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
    setup(sys, n);
    assign(sys);

    // 1. setup the system by calling assign()
    // 2. set the system
    // 3. measure
    switch (solver) {
        case pmcpp::Solver::TPR: {
            for (int i = 0; i < iter_times; i++) {
                assign(sys);
                TPR_CU::tpr_cu(sys->a, sys->c, sys->rhs, sys->diag, n, s);
            }
        } break;
        case pmcpp::Solver::PTPR: {
            for (int i = 0; i < iter_times; i++) {
                assign(sys);
                PTPR_CU::ptpr_cu(sys->a, sys->c, sys->rhs, sys->diag, n, s);
            }
        } break;
        case pmcpp::Solver::PCR: {
            for (int i = 0; i < iter_times; i++) {
                assign(sys);
                PTPR_CU::pcr_cu(sys->a, sys->c, sys->rhs, sys->diag, n);
            }
        } break;
        case pmcpp::Solver::cuSparse: {
            REFERENCE_CUSPARSE rfs(n);
            for (int i = 0; i < iter_times; i++) {
                assign(sys);
                rfs.solve(sys->a, sys->c, sys->rhs, sys->diag, n);
            }
        } break;
    }

    pmcpp::perf_time.display();

    clean(sys);
    free(sys);
}

int setup(struct TRIDIAG_SYSTEM *sys, int n) {
    sys->a = (real *)malloc(n * sizeof(real));
    sys->diag = (real *)malloc(n * sizeof(real));
    sys->c = (real *)malloc(n * sizeof(real));
    sys->rhs = (real *)malloc(n * sizeof(real));
    sys->n = n;

    return sys_null_check(sys);
}

int assign(struct TRIDIAG_SYSTEM *sys) {
    int n = sys->n;
    for (int i = 0; i < n; i++) {
        sys->a[i] = -1.0 / 6.0;
        sys->c[i] = -1.0 / 6.0;
        sys->diag[i] = 1.0;
        sys->rhs[i] = Random::get(-1., 1.);  // U(-1, 1)
    }
    sys->a[0] = 0.0;
    sys->c[n - 1] = 0.0;
    return 0;
}

int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p : {sys->a, sys->diag, sys->c, sys->rhs}) {
        if (p != nullptr) {
            free(p);
        }
    }

    sys->a = nullptr;
    sys->diag = nullptr;
    sys->c = nullptr;
    sys->rhs = nullptr;

    return 0;
}

bool sys_null_check(struct TRIDIAG_SYSTEM *sys) {
    for (auto p : {sys->a, sys->diag, sys->c, sys->rhs}) {
        if (p == nullptr) {
            return false;
        }
    }
    return true;
}