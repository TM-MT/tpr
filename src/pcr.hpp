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

        // TO-DO
        // make sure diag = {1., 1., ..., 1.};
    };

 
    int solve();

    int get_ans(real *x);

};
