#pragma once
#include "lib.hpp"

#ifdef PCR_SINGLE_THREAD
namespace PCRSingleThread {
#endif

class PCR : Solver {
    real *a, *c, *rhs;
    real *a1, *c1, *rhs1;
    int n;

   public:
    PCR(real *a, real *diag, real *c, real *rhs, int n) {
        init(n);
        set_tridiagonal_system(a, c, rhs);
    };

    PCR(int n) { init(n); }

    PCR(){};

    ~PCR() {
        delete[] this->a1;
        delete[] this->c1;
        delete[] this->rhs1;
    }

    PCR(const PCR &pcr) {
        n = pcr.n;
        init(pcr.n);
    };

    void init(int n) {
        this->n = n;

        this->a1 = new real[n];
        this->c1 = new real[n];
        this->rhs1 = new real[n];
    }

    void set_tridiagonal_system(real *a, real *diag, real *c, real *rhs) {
        if (diag != nullptr) {
            // set diag[i] = 1.0
            for (int i = 0; i < this->n; i++) {
                a[i] /= diag[i];
                c[i] /= diag[i];
                rhs[i] /= diag[i];
            }
        }

        set_tridiagonal_system(a, c, rhs);
    }

    void set_tridiagonal_system(real *a, real *c, real *rhs) {
        this->a = a;
        this->c = c;
        this->rhs = rhs;
    }

    int solve();

    int get_ans(real *x);
};

#ifdef PCR_SINGLE_THREAD
}
#endif
