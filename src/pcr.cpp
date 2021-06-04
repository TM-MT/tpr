#include <stdio.h>
#include <cmath>

#include "lib.hpp"
#include "pcr.hpp"


int PCR::solve() {
    int pn = (int)floor(log2((double)this->n));

    for (int p = 0; p < pn; p++) {
        int s = 1 << p;
        real a1[n], c1[n], rhs1[n];

        #pragma omp parallel shared(a1, c1, rhs1, s)
        {
            #pragma omp for
            for (int k = 0; k < n; k++) {
                int kl = max(k-s, 0);
                int kr = min(k+s, n-1);

                real ap = a[k];
                real cp = c[k];

                real e = 1.0 / ( 1.0 - ap * c[kl] - cp * a[kr] );

                a1[k] = -e * ap * a[kl];
                c1[k] = -e * cp * c[kr];
                rhs1[k] = e * ( rhs[k] - ap * rhs[kl] - cp * rhs[kr]);
            }

            #pragma omp for
            for (int k = 0; k < n; k++) {
                a[k] = a1[k];
                c[k] = c1[k];
                rhs[k] = rhs1[k];
            }
        }
    }

    return 0;
};

int PCR::get_ans(real *x) {
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        x[i] = this->rhs[i];
    }
    return 0;
};

