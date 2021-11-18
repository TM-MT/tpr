#include "pcr.hpp"

#include <stdio.h>

#include <cstring>

#include "lib.hpp"

#ifdef PCR_SINGLE_THREAD
namespace PCRSingleThread {
#endif

/**
 * @brief solve
 * @return num of float operation
 */
int PCR::solve() {
    int pn = fllog2(this->n);
    int n = this->n, margin = this->margin;

    for (int p = 0; p < pn - 1; p++) {
        int s = 1 << p;

#ifndef PCR_SINGLE_THREAD
#pragma omp parallel shared(a1, c1, rhs1) firstprivate(s, n, margin)
#endif
        {
#ifdef PCR_SINGLE_THREAD
#pragma omp simd
#else
#pragma omp for schedule(static)
#endif
            for (int k = margin; k < margin + n; k++) {
                int kl = k - s;
                int kr = k + s;
                assert(kl >= 0);
                assert(kr < n + 2 * margin);

                real ap = a[k];
                real cp = c[k];

                real e = 1.0f / (1.0f - ap * c[kl] - cp * a[kr]);

                int dst = k - margin;
                a1[dst] = -e * ap * a[kl];
                c1[dst] = -e * cp * c[kr];
                rhs1[dst] = e * (rhs[k] - ap * rhs[kl] - cp * rhs[kr]);
            }

#ifndef PCR_SINGLE_THREAD
#pragma omp for schedule(static)
#endif
            for (int k = 0; k < n; k++) {
                int dst = k + margin;
                a[dst] = a1[k];
                c[dst] = c1[k];
                rhs[dst] = rhs1[k];
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
        x[i] = this->rhs[i + margin];
    }
    return 0;
};

real *PCR::extend_input_array(real *p, int len) {
    real *ret = create_extend_array(len);
    int margin = array_margin(len);

    for (int i = 0; i < margin; i++) {
        ret[i] = 0.0;
        ret[2 * margin + n - i - 1] = 0.0;
    }
    std::memcpy(&ret[margin], p, len * sizeof(real));

    return ret;
}

void PCR::free_extend_input_array(real *p, int len) { delete[] p; }

/**
 * @brief      Creates an extend array for PCR_ESA use.
 *
 * @param[in]  oldlen  The oldlen
 *
 * @return     pointer `p` s.t. `p` points to `p[0]` and
 *             `p[-margin:oldlen+margin]`
 */
real *PCR::create_extend_array(int oldlen) {
    int margin = array_margin(oldlen);
    return create_extend_array(oldlen, margin);
}

/**
 * @brief      Creates an extend array for PCR_ESA use.
 *
 * @param[in]  oldlen  The oldlen
 * @param[in]  margin  The margin
 *
 * @return     pointer `p` s.t. `p` points to `p[0]` and
 *             `p[-margin:oldlen+margin]`
 */
real *PCR::create_extend_array(int oldlen, int margin) {
    int newlen = oldlen + 2 * margin;
    real *ret = new real[newlen];
    for (int i = 0; i < newlen; i++) {
        ret[i] = 0.0;
    }
    return ret;
}

#ifdef PCR_SINGLE_THREAD
}  // namespace PCRSingleThread
#endif
