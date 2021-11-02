#pragma once
#include "lib.hpp"



class REFERENCE_LAPACK : Solver {
    int n;
   public:
    REFERENCE_LAPACK(real *a, real *diag, real *c, real *rhs, int n) {
        init(n);
        set_tridiagonal_system(a, diag, c, rhs);
    };

    REFERENCE_LAPACK(int n) { init(n); }

    REFERENCE_LAPACK(){};

    ~REFERENCE_LAPACK() {
    }

    void set_tridiagonal_system(real *a, real *diag, real *c, real *rhs);

    int solve();

    int get_ans(real *x);

    /**
     * @brief Initialize CR with size `n`
     * @note call this before call any function in CR
     *
     * @param n size of the system
     */
    void init(int n) {
        this->n = n;
    }

   private:
    REFERENCE_LAPACK(const REFERENCE_LAPACK &cr);
    REFERENCE_LAPACK &operator=(const REFERENCE_LAPACK &cr);
};
