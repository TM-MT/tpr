#include "reference_lapack.hpp"

#include "lib.hpp"

/**
 * @brief set Tridiagnoal System
 * @note [OpenACC] given arrays are exists on the device
 *
 * @param a [description]
 * @param diag [description]
 * @param c [description]
 * @param rhs [description]
 */
void REFERENCE_LAPACK::set_tridiagonal_system(real *a, real *c, real *rhs) {
    this->dl = a;
    this->du = c;
    this->b = rhs;

    // d[0:n] = ones(0:n)
    for (int i = 0; i< this->n; i++) {
        this->d[i] = 1.0;
    }
}

int REFERENCE_LAPACK::solve() {
}


/**
 * @brief get the answer
 *
 * @note [OpenACC] assert `*x` exists at the device
 * @param x x[0:n]
 */
int REFERENCE_LAPACK::get_ans(real *x) {
}
