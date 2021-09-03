#pragma once
#include <lib.hpp>

class PCR: Solver
{
    real *a, *c, *rhs;
    real *a1, *c1, *rhs1;
    int n;

public:
    PCR(real __restrict *a, real __restrict *diag, __restrict real *c, real __restrict *rhs, int n) {
        this->a = a;
        this->c = c;
        this->rhs = rhs;
        this->n = n;

        this->a1 = new real[n];
        this->c1 = new real[n];
        this->rhs1 = new real[n];

        #pragma acc enter data copyin(this, this->n)
        #pragma acc enter data copyin(this->a[0:n], this->c[0:n], this->rhs[0:n])
        #pragma acc enter data create(a1[:n], c1[:n], rhs1[:n]) 

        // TO-DO
        // make sure diag = {1., 1., ..., 1.};
    };

    ~PCR() {
        delete[] this->a1;
        delete[] this->c1;
        delete[] this->rhs1;

        #pragma acc exit data delete(a1[:n], c1[:n], rhs1[:n])
        #pragma acc exit data copyout(this->a[0:n], this->c[0:n], this->rhs[0:n])
        #pragma acc exit data delete(this, this->n)
    }
 
    int solve();

    int get_ans(real *x);

};
