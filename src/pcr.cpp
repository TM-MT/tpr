#include "pcr.hpp"

#include <stdio.h>

#include <cstring>

#include "lib.hpp"

/**
 * @brief solve
 * @return num of float operation
 */
int PCR::solve() {
    int pn = fllog2(this->n);

    for (int p = 0; p < pn - 1; p++) {
        int s = 1 << p;

#pragma omp parallel shared(a1, c1, rhs1)
        {
#pragma omp for schedule(static)
            for (int k = this->margin; k < this->margin + this->n; k++) {
                int kl = k - s;
                int kr = k + s;
                assert(kl >= 0);
                assert(kr < this->n + 2 * this->margin);

                real ap = a[k];
                real cp = c[k];

                real e = 1.0 / (1.0 - ap * c[kl] - cp * a[kr]);

                int dst = k - this->margin;
                a1[dst] = -e * ap * a[kl];
                c1[dst] = -e * cp * c[kr];
                rhs1[dst] = e * (rhs[k] - ap * rhs[kl] - cp * rhs[kr]);
            }

#pragma omp for schedule(static)
            for (int k = 0; k < n; k++) {
                int dst = k + this->margin;
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
