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
    lapack_int n = this->n, nrhs = 1;
    this->info = gtsv(n, nrhs, this->dl, this->d, this->du, this->b, this->n);
    // from http://www.netlib.org/lapack/explore-html/d1/d88/group__real_g_tsolve_gae1cbb7cd9c376c9cc72575d472eba346.html#gae1cbb7cd9c376c9cc72575d472eba346
    // INFO is INTEGER
    // = 0: successful exit
    // < 0: if INFO = -i, the i-th argument had an illegal value
    // > 0: if INFO = i, U(i,i) is exactly zero, and the solution
    //    has not been computed.  The factorization has not been
    //    completed unless i = N.
    if (this->info != 0) { // if not successful
        char error_message[256];
        if (this->info < 0) {
            sprintf(error_message, "%d th argument had an illegal value.", -this->info);
        } else {
            sprintf(error_message, "U(%d, %d) is exactly zero, and the solution has not been computed.", this->info, this->info);
        }
        fprintf(stderr,
                "[Reference Lapack][Error] (error code: %d) `%s` at %s line %d\n", info, error_message, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

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
