#pragma once
#include <lib.hpp>

class PCR: Solver
{
    real *a, *diag, *c, *rhs;
    int n;

public:
    PCR(real *a, real *diag, real *c, real *rhs, int n) {
        this->a = a;
        this->diag = diag;
        this->c = c;
        this->rhs = rhs;
        this->n = n;

        // TO-DO
        // make sure diag = {1., 1., ..., 1.};
    };

 
    int solve();

    int get_ans(real *x);

};
