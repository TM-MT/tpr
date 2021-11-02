#include "reference_lapack.hpp"

#include "lib.hpp"
#include <lapacke.h>

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
    lapack_int info = 0;
    lapack_int n = this->n, nrhs = 1;
    info = gtsv(n, nrhs, this->dl, this->d, this->du, this->b, this->n);
    return static_cast<int>(info);
}


/**
 * @brief get the answer
 *
 * @note [OpenACC] assert `*x` exists at the device
 * @param x x[0:n]
 */
int REFERENCE_LAPACK::get_ans(real *x) {
}


/**
 * @brief      Helper function for {s|d}gstv
 *
 * @param[in]  n     { parameter_description }
 * @param[in]  nrhs  The nrhs
 * @param      dl    { parameter_description }
 * @param      d     { parameter_description }
 * @param      du    { parameter_description }
 * @param[in]  b     { parameter_description }
 * @param[in]  n     { parameter_description }
 *
 * @return     The lapack integer.
 */
lapack_int REFERENCE_LAPACK::gtsv(lapack_int n, lapack_int nrhs, real *dl, real *d, real *du, real *b, lapack_int ldb) {
    lapack_int info;
    #ifdef _REAL_IS_DOUBLE_
    info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n, nrhs, dl, d, du, b, ldb);
    #else
    info = LAPACKE_sgtsv(LAPACK_ROW_MAJOR, n, nrhs, dl, d, du, b, ldb);
    #endif
    return info;
}
