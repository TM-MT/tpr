#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>

#include "main.hpp"
#include "lib.hpp"
#include "cr.hpp"
#include "pcr.hpp"
#include "tpr.hpp"
#include "dbg.h"



int main() {
    int n = 1024;
    struct TRIDIAG_SYSTEM *sys = (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
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


    for (int s = 4; s <= n; s *= 2) {
        dbg(s);
        assign(sys);
        #pragma acc data copy(sys->a[:n], sys->c[:n], sys->rhs[:n], sys->n)
        {
            TPR t(sys->a, sys->diag, sys->c, sys->rhs, sys->n, s);
            t.solve();
            t.get_ans(sys->diag);
        }
        print_array(sys->diag, n);
        printf("\n");
    }

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
        sys->rhs[i] = 1.0 * (i+1);
    }
    sys->a[0] = 0.0;
    sys->c[n-1] = 0.0;

    return 0;
}



int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p: { sys->a, sys->diag, sys->c, sys->rhs }) {
        free(p);
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
