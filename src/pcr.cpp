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
        #pragma acc kernels create(a1[:n], c1[:n], rhs1[:n]) present(this->a[-margin:n+margin], this->c[-margin:n+margin], this->rhs[-margin:n+margin], this, this->n)
        #ifdef _OPENMP
        #pragma omp parallel shared(a1, c1, rhs1)
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
            for (int k = 0; k < n; k++) {
                int kl = k - s;
                int kr = k + s;

                real ap = a[k];
                real cp = c[k];

                real e = 1.0 / ( 1.0 - ap * c[kl] - cp * a[kr] );

                a1[k] = -e * ap * a[kl];
                c1[k] = -e * cp * c[kr];
                rhs1[k] = e * ( rhs[k] - ap * rhs[kl] - cp * rhs[kr]);
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

real* PCR::extend_input_array(real *p, int len) {
    int margin = array_margin(len);
    real* start_point = extend_array(p, len, len + 2 * margin, margin);
    return &start_point[margin];
}

real* PCR::extend_array(real *p, int oldlen, int newlen, int margin) {
    real *ret = new real[newlen];

    // fill 0
    for (int i = 0; i < newlen; i++) {
        ret[i] = 0.0;
    }

    // copy
    for (int i = 0; i < oldlen; i++) {
        ret[margin + i] = p[i];
    }

    return ret;
}

