#include <stdio.h>
#include "lib.hpp"
#include "pcr.hpp"


/**
 * @brief solve
 * @return num of float operation
 */
int PCR::solve() {
    int pn = fllog2(this->n);

    for (int p = 0; p < pn; p++) {
        int s = 1 << p;
        real a1[n], c1[n], rhs1[n];

        #pragma omp parallel shared(a1, c1, rhs1, s)
        {
            #pragma omp for schedule(static)
            for (int k = 0; k < s; k++) {
                int kr = k + s;

                real ap = a[k];
                real cp = c[k];

                real e = 1.0 / ( 1.0 - cp * a[kr] );

                a1[k] = e * ap;
                c1[k] = -e * cp * c[kr];
                rhs1[k] = e * ( rhs[k] - cp * rhs[kr]);
            }

            #pragma omp for schedule(static)
            for (int k = s; k < n - s; k++) {
                int kl = k - s;
                int kr = k + s;

                real ap = a[k];
                real cp = c[k];

                real e = 1.0 / ( 1.0 - ap * c[kl] - cp * a[kr] );

                a1[k] = -e * ap * a[kl];
                c1[k] = -e * cp * c[kr];
                rhs1[k] = e * ( rhs[k] - ap * rhs[kl] - cp * rhs[kr]);
            }

            #pragma omp for schedule(static)
            for (int k = n - s; k < n; k++) {
                int kl = k - s;

                real ap = a[k];
                real cp = c[k];

                real e = 1.0 / ( 1.0 - ap * c[kl] );

                a1[k] = -e * ap * a[kl];
                c1[k] = e * cp;
                rhs1[k] = e * ( rhs[k] - ap * rhs[kl]);
            }

            for (int k = 0; k < n; k++) {
                a[k] = a1[k];
                c[k] = c1[k];
                rhs[k] = rhs1[k];
            }
        }
    }

    return 14 * n * pn;
};

/**
 * @brief get the answer
 * @return num of float operation
 */
int PCR::get_ans(real *x) {
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        x[i] = this->rhs[i];
    }
    return 0;
};

