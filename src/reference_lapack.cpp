#include "reference_lapack.hpp"

#include <lapacke.h>

#include <cmath>

#include "lib.hpp"

/**
 * @brief set Tridiagnoal System
 *
 * @param a [description]
 * @param diag [description]
 * @param c [description]
 * @param rhs [description]
 */
void REFERENCE_LAPACK::set_tridiagonal_system(real *a, real *diag, real *c,
                                              real *rhs) {
    this->dl = a;
    this->d = diag;
    this->du = c;
    this->b = rhs;
}

/**
 * @brief set Tridiagnoal System
 * @note call this may overwrite `this->d`
 *
 * @param a [description]
 * @param c [description]
 * @param rhs [description]
 */
void REFERENCE_LAPACK::set_tridiagonal_system(real *a, real *c, real *rhs) {
    this->dl = a;
    this->diag_allocated = true;
    delete[] this->d;
    this->d = new real[this->n];
    this->du = c;
    this->b = rhs;

    // d[0:n] = ones(0:n)
    for (int i = 0; i < this->n; i++) {
        this->d[i] = 1.0f;
    }
}

int REFERENCE_LAPACK::solve() {
    lapack_int n = this->n, nrhs = 1, ldb = 1;
    this->info = gtsv(n, nrhs, this->dl, this->d, this->du, this->b, ldb);
    // from
    // http://www.netlib.org/lapack/explore-html/d1/d88/group__real_g_tsolve_gae1cbb7cd9c376c9cc72575d472eba346.html#gae1cbb7cd9c376c9cc72575d472eba346
    // INFO is INTEGER
    // = 0: successful exit
    // < 0: if INFO = -i, the i-th argument had an illegal value
    // > 0: if INFO = i, U(i,i) is exactly zero, and the solution
    //    has not been computed.  The factorization has not been
    //    completed unless i = N.
    //
    // when INFO < 0 -> show message and exit
    // when INFO > 0 -> show message ONLY
    if (this->info != 0) {  // if not successful
        char error_message[256];
        if (this->info < 0) {
            sprintf(error_message, "%d th argument had an illegal value.",
                    -this->info);
        } else {
            sprintf(error_message,
                    "U(%d, %d) is exactly zero, and the solution has not been "
                    "computed.",
                    this->info, this->info);
        }
        fprintf(
            stderr,
            "[Reference Lapack][Error] (error code: %d) `%s` at %s line %d\n",
            info, error_message, __FILE__, __LINE__);

        // when `info < 0`, this is fatal error
        if (this->info < 0) {
            exit(EXIT_FAILURE);
        }
    }

    return static_cast<int>(info);
}

/**
 * @brief get the answer
 *
 * @param x x[0:n]
 */
int REFERENCE_LAPACK::get_ans(real *x) {
    if (this->info == 0) {  // On successful exit
        for (int i = 0; i < n; i++) {
            x[i] = this->b[i];
        }
    } else if (this->info > 0) {  // the solution has not been computed
        for (int i = 0; i < n; i++) {
            x[i] = NAN;
        }
    } else {
        assert(false);
    }

    this->info = -1;
    return 0;
}

/**
 * @brief      Helper function for {s|d}gstv
 *
 * see
 * http://www.netlib.org/lapack/explore-html/d1/d88/group__real_g_tsolve_gae1cbb7cd9c376c9cc72575d472eba346.html#gae1cbb7cd9c376c9cc72575d472eba346
 * for more information
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
lapack_int REFERENCE_LAPACK::gtsv(lapack_int n, lapack_int nrhs, real *dl,
                                  real *d, real *du, real *b, lapack_int ldb) {
    lapack_int info;
#ifdef _REAL_IS_DOUBLE_
    info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n, nrhs, dl, d, du, b, ldb);
#else
    info = LAPACKE_sgtsv(LAPACK_ROW_MAJOR, n, nrhs, dl, d, du, b, ldb);
#endif
    return info;
}
