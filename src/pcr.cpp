#include <stdio.h>
#include "lib.hpp"
#include "pcr.hpp"


/**
 * @brief solve
 * @return num of float operation
 */
int PCR::solve() {
    int pn = fllog2(this->n);
    real a1[n], c1[n], rhs1[n];

    for (int p = 0; p < pn; p++) {
        #pragma acc kernels create(a1[:n], c1[:n], rhs1[:n]) present(this->a[:n], this->c[:n], this->rhs[:n], this, this->n)
        #ifdef _OPENMP
        #pragma omp parallel shared(a1, c1, rhs1, s)
        #endif
        {
            #pragma acc update device(p)
            int s = 1 << p;

            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int k = 0; k < s; k++) {
                int kr = k + s;

                real e = 1.0 / ( 1.0 - c[k] * a[kr]);

                a1[k] = e * a[k];
                c1[k] = -e * c[k] * c[kr];
                rhs1[k] = e * (rhs[k] - c[k] * rhs[kr]);
            }

            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
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

            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int k = n - s; k < n; k++) {
                int kl = k - s;

                real ap = a[k];
                real cp = c[k];

                real e = 1.0 / ( 1.0 - ap * c[kl] );

                a1[k] = -e * ap * a[kl];
                c1[k] = e * cp;
                rhs1[k] = e * ( rhs[k] - ap * rhs[kl]);
            }

            #pragma acc loop
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
    #pragma acc update host(this->rhs[:this->n]) wait

    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for (int i = 0; i < n; i++) {
        x[i] = this->rhs[i];
    }
    return 0;
};

