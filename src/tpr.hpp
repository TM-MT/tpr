#pragma once
#include <assert.h>
#include <cmath>

#include "lib.hpp"


class TPR: Solver
{
    real *a, *diag, *c, *rhs, *x;
    int n, s;

public:
    TPR(real *a, real *diag, real *c, real *rhs, int n, int s) {
        this->a = a;
        this->diag = diag;
        this->c = c;
        this->rhs = rhs;
        this->n = n;
        this->s = s;

        this->x = (real *)malloc(sizeof(real) * n);

        assert(floor((double)n / s) == ceil((double)n / s));
        assert(4 <= s && s <= n);
    };
 
    int solve();

    int get_ans(real *x);

private:
    void tpr_forward();

    void update_section(int i, int u);
    void update_global(int i, int u);
    void update_bd_check(int i, int u, int lb, int ub);


    void update_no_check(int kl, int k, int kr);
    void update_uppper_no_check(int i, int kr);
    void update_lower_no_check(int kl, int i);

    void tpr_backward();

    void replace(int i, int j);

    void tpr_stage1(int st, int ed);

};
