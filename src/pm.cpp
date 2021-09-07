#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <string>

#include "PerfMonitor.h"
#include "main.hpp"
#include "lib.hpp"
#include "pcr.hpp"
#include "tpr.hpp"
#include "effolkronium/random.hpp"
#include "dbg.h"
#include <structopt/app.hpp>
#include "pm.hpp"


// std::mt19937 base pseudo-random
using Random = effolkronium::random_static;


/**
 * @brief Command Line Args
 * @details Define Command Line Arguments
 */
struct Options {
    // size of system
    std::optional<int> n = 2048;
    // size of slice
    std::optional<int> s = 1024;
    // Iteration Times
    std::optional<int> iter = 1000;
    // Solver
    std::optional<std::string> solver = "TPR";
};
STRUCTOPT(Options, n, s, iter, solver);

pm_lib::PerfMonitor pmcpp::pm = pm_lib::PerfMonitor();


pmcpp::Solver pmcpp::str2Solver(std::string &solver) {
    to_lower(solver);
    if (solver.compare(std::string("pcr")) == 0) {
        return Solver::PCR;
    } else if (solver.compare(std::string("tpr")) == 0) {
        return Solver::TPR;
    } else{
        std::cerr << "Solver Not Found.\n";
        abort();
    }
}

void pmcpp::to_lower(std::string &s1) {
   std::transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
}


int main(int argc, char *argv[]) {
    int n, s, iter_times;
    pmcpp::Solver solver;
    // Parse Command Line Args
    try {
        auto options = structopt::app("tpr_pm", "v1.0.0").parse<Options>(argc, argv);
        n = options.n.value();
        s = options.s.value();
        iter_times = options.iter.value();
        solver = pmcpp::str2Solver(options.solver.value());
    } catch (structopt::exception& e) {
        std::cout << e.what() << "\n";
        std::cout << e.help();
        exit(EXIT_FAILURE);
    }

    // print type infomation
    {
        const real v = 0.0;
        std::cerr << "type info: " ;
        dbg(v);
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
