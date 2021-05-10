#pragma once

#include "lib.hpp"


struct TRIDIAG_SYSTEM
{
    int n;
    real *a;
    real *diag;
    real *c;
    real *rhs;
};


int setup(struct TRIDIAG_SYSTEM *sys, int n);
int clean(struct TRIDIAG_SYSTEM *sys);
int assign(struct TRIDIAG_SYSTEM *sys);
bool sys_null_check(struct TRIDIAG_SYSTEM *sys);
