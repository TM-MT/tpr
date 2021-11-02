#pragma once
#include "lib.hpp"
#include <lapacke.h>

/**
 * @brief      x = (real *)malloc(sizeof(real) * n)
 *
 * @param      x     *real
 * @param      n     length of array
 *
 */
#define RMALLOC(x, n) x = new real[n]

/**
 * @brief Safely delete pointer `p` and set `p = nullptr`
 */
#define SAFE_DELETE(p) \
    delete[] p;        \
    p = nullptr


class REFERENCE_LAPACK : Solver {
    int n;
    real *dl, *d, *du, *b;

   public:
    REFERENCE_LAPACK(real *a, real *diag, real *c, real *rhs, int n) {
        init(n);
        set_tridiagonal_system(a, c, rhs);
    };

    REFERENCE_LAPACK(int n) { init(n); }

    REFERENCE_LAPACK(){};

    ~REFERENCE_LAPACK() {
        SAFE_DELETE(this->dl);
        SAFE_DELETE(this->d);
        SAFE_DELETE(this->du);
        SAFE_DELETE(this->b);
    }

    void set_tridiagonal_system(real *a, real *c, real *rhs);

    int solve();

    int get_ans(real *x);

    /**
     * @brief Initialize REFERENCE_LAPACK with size `n`
     * @note call this before call any function in CR
     *
     * @param n size of the system
     */
    void init(int n) {
        this->n = n;

        RMALLOC(this->dl, n);
        RMALLOC(this->d, n);
        RMALLOC(this->du, n);
        RMALLOC(this->b, n);
    }

   private:
    REFERENCE_LAPACK(const REFERENCE_LAPACK &cr);
    REFERENCE_LAPACK &operator=(const REFERENCE_LAPACK &cr);

    lapack_int gtsv(lapack_int n, lapack_int nrhs, real *dl, real *d, real *du, real *b, lapack_int ldb);
};

#undef RMALLOC
#undef SAFE_DELETE
