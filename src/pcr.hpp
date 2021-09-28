#pragma once
#include "lib.hpp"

class PCR : Solver {
    real *a, *c, *rhs;
    real *a1, *c1, *rhs1;
    int n;

   public:
    PCR(real *a, real *diag, real *c, real *rhs, int n) {
        init(n);
        set_tridiagonal_system(a, diag, c, rhs);
    };

    PCR(){};

    ~PCR() {
        delete[] this->a1;
        delete[] this->c1;
        delete[] this->rhs1;

#pragma acc exit data detach(this->a, this->c, this->rhs)
#pragma acc exit data delete (a1[:n], c1[:n], rhs1[:n])
#pragma acc exit data delete (this)
    }

    void init(int n) {
        this->n = n;

        this->a1 = new real[n];
        this->c1 = new real[n];
        this->rhs1 = new real[n];

#pragma acc enter data copyin(this)
#pragma acc enter data create(a1[:n], c1[:n], rhs1[:n])
    }

    void set_tridiagonal_system(real *a, real *diag, real *c, real *rhs) {
        this->a = a;
        this->c = c;
        this->rhs = rhs;

#pragma acc update device(this)
#pragma acc enter data attach(this->a, this->c, this->rhs)
    }

    int solve();

    int get_ans(real *x);

   private:
    PCR(const PCR &pcr);
    PCR &operator=(const PCR &pcr);
};
