#pragma once
#include <assert.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include "lib.hpp"

/**
 * @brief      x = (real *)malloc(sizeof(real) * n)
 *
 * @param      x     *real
 * @param      n     length of array
 *
 */
#define RMALLOC(x, n) x = (real *)malloc(sizeof(real) * n)

/**
 * @brief      Infomation of equation and its index
 */
struct EquationInfo {
    int idx;
    real a;
    real c;
    real rhs;
};

class TPR: Solver
{
    real *a, *c, *rhs, *x;
    real *init_a, *init_c, *init_rhs;
    real *st1_a, *st1_c, *st1_rhs;
    int n, s;

public:
    TPR(real *a, real *diag, real *c, real *rhs, int n, int s) {
        this->a = a;
        // assert diag == { 1.0, 1.0, ... 1.0 }
        this->c = c;
        this->rhs = rhs;
        this->n = n;
        this->s = s;

        // allocation for answer
        RMALLOC(this->x, n);

        // allocation for backup
        RMALLOC(this->init_a, (n + 1) / 2);
        RMALLOC(this->init_c, (n + 1) / 2);
        RMALLOC(this->init_rhs, (n + 1) / 2);
        RMALLOC(this->st1_a, (n + 1) / 2);
        RMALLOC(this->st1_c, (n + 1) / 2);
        RMALLOC(this->st1_rhs, (n + 1) / 2);

        // NULL CHECK
        real **ps[] = { &this->init_a, &this->init_c, &this->init_rhs,
                        &this->st1_a, &this->st1_c, &this->st1_rhs,
        };
        for (int i = 0; i < sizeof(ps) / sizeof(ps[0]); i++) {
            if (ps[i] == NULL) {
                printf("[%s] FAILED TO ALLOCATE %d th array.\n",
                    __func__, i
                    );
                abort();
            }
        }

        // Initialize the answer array
        for (int i = 0; i < n; i++) {
            this->x[i] = 0.0;
        }
        assert(floor((double)n / s) == ceil((double)n / s));
        assert(4 <= s && s <= n);
    };
 
    int solve();

    int get_ans(real *x);

private:
    EquationInfo update_section(int i, int u);
    EquationInfo update_global(int i, int u);
    EquationInfo update_bd_check(int i, int u, int lb, int ub);

    void mk_bkup_init(int st, int ed);
    void mk_bkup_st1(int st, int ed);
    void bkup_cp(real *src, real *dst, int st,int ed);

    EquationInfo update_no_check(int kl, int k, int kr);
    EquationInfo update_uppper_no_check(int k, int kr);
    EquationInfo update_lower_no_check(int kl, int k);

    void replace_with_init(int i);
    void replace_with_st1(int i);

    void tpr_stage1(int st, int ed);
    void tpr_stage2();
    void tpr_stage3(int st, int ed);

    void patch_equation_info(EquationInfo eqi);
};
