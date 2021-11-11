#pragma once
#include "lib.hpp"

class PCR : Solver {
    real *a, *c, *rhs;
    real *a1, *c1, *rhs1;
    int n;
    int margin;

   public:
    PCR(real *a, real *diag, real *c, real *rhs, int n) {
        init(n);
        set_tridiagonal_system(a, diag, c, rhs);
    };

    PCR(){};

    ~PCR() {
        delete[] & this->a[-this->margin];
        delete[] & this->c[-this->margin];
        delete[] & this->rhs[-this->margin];
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
        this->margin = array_margin(n);

        this->a1 = new real[n];
        this->c1 = new real[n];
        this->rhs1 = new real[n];
    }

    void set_tridiagonal_system(real *a, real *diag, real *c, real *rhs) {
        this->a = extend_input_array(a, n);
        this->c = extend_input_array(c, n);
        this->rhs = extend_input_array(rhs, n);
    }

    int solve();

    int get_ans(real *x);

    real *extend_input_array(real *p, int len);

   private:
    real *extend_array(real *p, int oldlen, int newlen, int margin);
    inline int array_margin(int n) { return 1 << fllog2(n); }
};
