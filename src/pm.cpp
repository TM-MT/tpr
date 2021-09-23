#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <string>

#include "PerfMonitor.h"
#include "main.hpp"
#include "lib.hpp"
#include "pcr.hpp"
#include "tpr.hpp"
#include "ptpr.hpp"
#include "effolkronium/random.hpp"
#include "pm.hpp"


// std::mt19937 base pseudo-random
using Random = effolkronium::random_static;


namespace pmcpp {
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

    void to_lower(std::string &s1);
    Solver str2Solver(std::string &solver);


    Solver str2Solver(std::string solver) {
        to_lower(solver);
        if (solver.compare(std::string("pcr")) == 0) {
            return Solver::PCR;
        } else if (solver.compare(std::string("tpr")) == 0) {
            return Solver::TPR;
        } else if (solver.compare(std::string("ptpr")) == 0) {
            return Solver::PTPR;
        } else{
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
}


pm_lib::PerfMonitor pmcpp::pm = pm_lib::PerfMonitor();



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

    struct TRIDIAG_SYSTEM *sys = (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
    setup(sys, n);
    assign(sys);

    pmcpp::pm.initialize(100);

    // 1. setup the system by calling assign()
    // 2. set the system
    // 3. measure
    switch (solver) {
        case pmcpp::Solver::TPR: {
            auto tpr_all_label = std::string("TPR_").append(std::to_string(s));
            pmcpp::pm.setProperties(tpr_all_label, pmcpp::pm.CALC);

            // Measureing TPR reusable implementation
            {
                TPR t(sys->n, s);
                for (int i = 0; i < iter_times; i++) {
                    assign(sys);
                    #pragma acc data copy(sys->a[:n], sys->diag[:n], sys->c[:n], sys->rhs[:n], sys->n)
                    {
                        t.set_tridiagonal_system(sys->a, sys->c, sys->rhs);
                        pmcpp::pm.start(tpr_all_label);
                        int flop_count = t.solve();
                        flop_count += t.get_ans(sys->diag);
                        pmcpp::pm.stop(tpr_all_label, flop_count);
                    }
                }
            }
        } break;
        case pmcpp::Solver::PTPR: {
            auto tpr_all_label = std::string("PTPR_").append(std::to_string(s));
            pmcpp::pm.setProperties(tpr_all_label, pmcpp::pm.CALC);

            // Measureing TPR reusable implementation
            {
                PTPR t(sys->n, s);
                for (int i = 0; i < iter_times; i++) {
                    assign(sys);
                    #pragma acc data copy(sys->a[:n], sys->diag[:n], sys->c[:n], sys->rhs[:n], sys->n)
                    {
                        t.set_tridiagonal_system(sys->a, sys->c, sys->rhs);
                        pmcpp::pm.start(tpr_all_label);
                        int flop_count = t.solve();
                        flop_count += t.get_ans(sys->diag);
                        pmcpp::pm.stop(tpr_all_label, flop_count);
                    }
                }
            }
        } break;
        case pmcpp::Solver::PCR: {
            auto pcr_label = std::string("PCR");
            pmcpp::pm.setProperties(pcr_label);
            for (int i = 0; i < iter_times; i++) {
                assign(sys);

                #pragma acc data copy(sys->a[:n], sys->diag[:n], sys->c[:n], sys->rhs[:n], sys->n)
                {
                    PCR p(sys->a, sys->diag, sys->c, sys->rhs, sys->n);
                    pmcpp::pm.start(pcr_label);
                    int flop_count = p.solve();
                    flop_count += p.get_ans(sys->diag);
                    pmcpp::pm.stop(pcr_label, flop_count);
                }
            }
        } break;
    }

    pmcpp::pm.print(stdout, std::string(""), std::string(), 1);
    pmcpp::pm.printDetail(stdout, 0, 1);

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
        sys->a[i] = -1.0/6.0;
        sys->c[i] = -1.0/6.0;
        sys->diag[i] = 1.0;
        sys->rhs[i] = Random::get(-1., 1.);  // U(-1, 1)
    }
    sys->a[0] = 0.0;
    sys->c[n-1] = 0.0;
    return 0;
}



int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p: { sys->a, sys->diag, sys->c, sys->rhs }) {
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
    for (auto p: { sys->a, sys->diag, sys->c, sys->rhs }) {
        if (p == nullptr) {
            return false;
        }
    }
    return true;
}
