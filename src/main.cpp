#include "main.hpp"

#include <stdio.h>
#include <stdlib.h>

#include <initializer_list>

#include "PerfMonitor.h"
#include "cr.hpp"
#include "lib.hpp"
#include "pcr.hpp"
#include "pm.hpp"
#include "ptpr.hpp"
#include "tpr.hpp"

pm_lib::PerfMonitor pmcpp::pm = pm_lib::PerfMonitor();

int main() {
    int n = 1024;
    struct TRIDIAG_SYSTEM *sys =
        (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
    setup(sys, n);

    assign(sys);
#pragma acc data copy(sys->a[:n], sys->c[:n], sys->rhs[:n], sys->n)
    {
        CR cr(sys->a, sys->diag, sys->c, sys->rhs, sys->n);
        cr.solve();
        cr.get_ans(sys->diag);
    }
    print_array(sys->diag, n);
    printf("\n");

    assign(sys);
#pragma acc data copy(sys->a[:n], sys->c[:n], sys->rhs[:n], sys->n)
    {
        PCR pcr(sys->a, sys->diag, sys->c, sys->rhs, sys->n);
        pcr.solve();
        pcr.get_ans(sys->diag);
    }
    print_array(sys->diag, n);
    printf("\n");

    pmcpp::pm.initialize(100);
    auto tpr_label = std::string("TPR");
    pmcpp::pm.setProperties(tpr_label, pmcpp::pm.CALC);

    for (int s = 4; s <= n; s *= 2) {
        std::cerr << "TPR s=" << s << "\n";
        assign(sys);
#pragma acc data copy( \
    sys->a[:n], sys->diag[:n], sys->c[:n], sys->rhs[:n], sys->n)
        {
            TPR t(sys->a, sys->diag, sys->c, sys->rhs, sys->n, s);
            pmcpp::pm.start(tpr_label);
            int flop_count = t.solve();
            flop_count += t.get_ans(sys->diag);
            pmcpp::pm.stop(tpr_label, flop_count);
        }
        print_array(sys->diag, n);
        printf("\n");
    }

    tpr_label = std::string("PTPR");
    pmcpp::pm.setProperties(tpr_label, pmcpp::pm.CALC);

    for (int s = 4; s <= n; s *= 2) {
        std::cerr << "PTPR s=" << s << "\n";
        assign(sys);
#pragma acc data copy( \
    sys->a[:n], sys->diag[:n], sys->c[:n], sys->rhs[:n], sys->n)
        {
            PTPR t(sys->a, sys->diag, sys->c, sys->rhs, sys->n, s);
            pmcpp::pm.start(tpr_label);
            int flop_count = t.solve();
            flop_count += t.get_ans(sys->diag);
            pmcpp::pm.stop(tpr_label, flop_count);
        }
        print_array(sys->diag, n);
        printf("\n");
    }

    pmcpp::pm.print(stderr, std::string(""), std::string(), 1);
    pmcpp::pm.printDetail(stderr, 0, 1);

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
        sys->rhs[i] = 1.0 * (i + 1);
    }
    sys->a[0] = 0.0;
    sys->c[n - 1] = 0.0;

    return 0;
}

int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p : {sys->a, sys->diag, sys->c, sys->rhs}) {
        free(p);
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
