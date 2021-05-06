#include <stdio.h>
#include <stdlib.h>
#include <initializer_list>

#include "main.hpp"
#include "lib.hpp"
#include "pcr.hpp"



int main() {
    real *a, *diag, *c, *rhs;
    int n = 8;

    a = (real *)malloc(n * sizeof(real));
    diag = (real *)malloc(n * sizeof(real));
    c = (real *)malloc(n * sizeof(real));
    rhs = (real *)malloc(n * sizeof(real));

    for (int i = 0; i < n; i++) {
        a[i] = 1.0/6.0;
        c[i] = 1.0/6.0;
        diag[i] = 1.0;
        rhs[i] = 1.0 * i;
    }
    a[0] = 0.0;
    c[n-1] = 0.0;

    auto pcr = PCR(a, diag, c, rhs, n);
    pcr.solve();
    pcr.get_ans(diag);

    print_array(diag, n);

    for (auto p: { a, diag, c, rhs }) {
        if (p != NULL) {
            free(p);
        }
    }
}
