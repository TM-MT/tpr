#include "pcr.hpp"

#include <stdio.h>

#include "lib.hpp"
#include "omp.h"

#ifdef PCR_SINGLE_THREAD
namespace PCRSingleThread {
#endif

/**
 * @brief solve
 * @return num of float operation
 */
int PCR::solve() {
    int pn = fllog2(this->n);

    for (int p = 0; p < pn - 1; p++) {
        int s = 1 << p;

#ifndef PCR_SINGLE_THREAD
#pragma omp parallel shared(a1, c1, rhs1)
#endif
        {
#ifdef PCR_SINGLE_THREAD
#pragma omp simd
#else
#pragma omp for schedule(static)
#endif
            for (int k = 0; k < s; k++) {
                int kr = k + s;

                real e = 1.0f / (1.0f - c[k] * a[kr]);

                a1[k] = e * a[k];
                c1[k] = -e * c[k] * c[kr];
                rhs1[k] = e * (rhs[k] - c[k] * rhs[kr]);
            }

#ifdef PCR_SINGLE_THREAD
#pragma omp simd
#else
#pragma omp for schedule(static)
#endif
            for (int k = s; k < n - s; k++) {
                int kl = k - s;
                int kr = k + s;

                real ap = a[k];
                real cp = c[k];

                real e = 1.0f / (1.0f - ap * c[kl] - cp * a[kr]);

                a1[k] = -e * ap * a[kl];
                c1[k] = -e * cp * c[kr];
                rhs1[k] = e * (rhs[k] - ap * rhs[kl] - cp * rhs[kr]);
            }

#ifdef PCR_SINGLE_THREAD
#pragma omp simd
#else
#pragma omp for schedule(static)
#endif
            for (int k = n - s; k < n; k++) {
                int kl = k - s;

                real ap = a[k];
                real cp = c[k];

                real e = 1.0f / (1.0f - ap * c[kl]);

                a1[k] = -e * ap * a[kl];
                c1[k] = e * cp;
                rhs1[k] = e * (rhs[k] - ap * rhs[kl]);
            }

#ifndef PCR_SINGLE_THREAD
#pragma omp for schedule(static)
#endif
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
 *
 * @return num of float operation
 */
int PCR::get_ans(real *x) {
    for (int i = 0; i < n; i++) {
        x[i] = this->rhs[i];
    }
    return 0;
};

#ifdef PCR_SINGLE_THREAD
}  // namespace PCRSingleThread
#endif
