#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>
#include <PerfMonitor.h>
#include <string>

#include "main.hpp"
#include "pm.hpp"
#include "lib.hpp"
#include "pcr.hpp"
#include "tpr.hpp"
#include "effolkronium/random.hpp"
#include "dbg.h"
#include <structopt/app.hpp>


pm_lib::PerfMonitor PM;

// std::mt19937 base pseudo-random
using Random = effolkronium::random_static;


/**
 * @brief Command Line Args
 * @details Define Command Line Arguments
 */
struct Options {
    // size of system
    std::optional<int> n = 2048;
    // Iteration Times
    std::optional<int> iter = 1000;
};
STRUCTOPT(Options, n, iter);

int main(int argc, char *argv[]) {
    int n, iter_times;
    // Parse Command Line Args
    try {
        auto options = structopt::app("tpr_pm", "v1.0.0").parse<Options>(argc, argv);
        n = options.n.value();
        iter_times = options.iter.value();
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

    // Initialize PerfMonitor and set labels
    PM.initialize(100);
    PM.setProperties(std::string("PCR"), PM.CALC);
    for (int s = 4; s <= n; s *= 2) {
        auto label = std::string("TPR", 3);
        label.append(std::to_string(s));
        PM.setProperties(label, PM.CALC);
    }
    {
        // assert n is power of 2
        std::string label = std::string("TPR");
        label.append(std::to_string(n / 2));
        label.append(std::string("_Reusable"));
        PM.setProperties(label, PM.CALC);
    }


    struct TRIDIAG_SYSTEM *sys = (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
    setup(sys, n);
    assign(sys);

    // Measuring PCR
    for (int i = 0; i < iter_times; i++) {
        PM.start(std::string("PCR", 3));
        PCR p = PCR(sys->a, sys->diag, sys->c, sys->rhs, sys->n);
        int flop_count = p.solve();
        flop_count += p.get_ans(sys->diag);
        PM.stop(std::string("PCR", 3), (double)flop_count, 1);
    }

    // Measureing TPR
    for (int s = 4; s <= n; s *= 2) {
        for (int i = 0; i < iter_times; i++) {
            assign(sys);
            auto label = std::string("TPR", 3).append(std::to_string(s));
            PM.start(label);
            TPR t = TPR(sys->a, sys->diag, sys->c, sys->rhs, sys->n, s);
            int flop_count = t.solve();
            flop_count += t.get_ans(sys->diag);
            PM.stop(label, (double)flop_count, 1);
        }
    }


    // Measureing TPR reusable implementation
    {
        int s = n / 2;
        TPR t = TPR(sys->n, s);
        for (int i = 0; i < iter_times; i++) {
            assign(sys);
            t.set_tridiagonal_system(sys->a, sys->c, sys->rhs);
            auto label = std::string("TPR", 3)
                            .append(std::to_string(s))
                            .append(std::string("_Reusable"));
            PM.start(label);
            int flop_count = t.solve();
            flop_count += t.get_ans(sys->diag);
            PM.stop(label, (double)flop_count, 1);
        }
    }

    PM.print(stdout, std::string(""), std::string(), 1);
    PM.printDetail(stdout, 0, 1);

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
