#pragma once
#include <lib.hpp>

class PCR: Solver
{
    real *a, *c, *rhs;
    int n;

public:
    PCR(real __restrict *a, real __restrict *diag, __restrict real *c, real __restrict *rhs, int n) {
        this->a = a;
        this->c = c;
        this->rhs = rhs;
        this->n = n;

        #pragma acc enter data create(this, this->n)
        #pragma acc enter data copyin(this->a[0:n], this->c[0:n], this->rhs[0:n]) wait
        // TO-DO
        // make sure diag = {1., 1., ..., 1.};
    };

    ~PCR() {
        #pragma acc exit data copyout(this->a[0:n], this->c[0:n], this->rhs[0:n])
        #pragma acc exit data delete(this, this->n)
    }
 
    int solve();

    int get_ans(real *x);

};
