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
    real diag;
    real c;
    real rhs;
};

class TPR: Solver
{
    real *a, *diag, *c, *rhs, *x;
    real *init_a, *init_diag, *init_c, *init_rhs;
    real *st1_a, *st1_diag, *st1_c, *st1_rhs;
    int n, s;

public:
    TPR(real *a, real *diag, real *c, real *rhs, int n, int s) {
        this->a = a;
        this->diag = diag;
        this->c = c;
        this->rhs = rhs;
        this->n = n;
        this->s = s;

        // allocation for answer
        RMALLOC(this->x, n);

        // allocation for backup
        RMALLOC(this->init_a, (n + 1) / 2);
        RMALLOC(this->init_diag, (n + 1) / 2);
        RMALLOC(this->init_c, (n + 1) / 2);
        RMALLOC(this->init_rhs, (n + 1) / 2);
        RMALLOC(this->st1_a, (n + 1) / 2);
        RMALLOC(this->st1_diag, (n + 1) / 2);
        RMALLOC(this->st1_c, (n + 1) / 2);
        RMALLOC(this->st1_rhs, (n + 1) / 2);

        // NULL CHECK
        real **ps[] = { &this->init_a, &this->init_diag, 
                        &this->init_c, &this->init_rhs, 
                        &this->st1_a, &this->st1_diag, 
                        &this->st1_c, &this->st1_rhs,
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
    void tpr_forward();

    EquationInfo update_section(int i, int u);
    EquationInfo update_global(int i, int u);
    EquationInfo update_bd_check(int i, int u, int lb, int ub);

    void mk_bkup_init();
    void mk_bkup_st1();
    void bkup_cp(real *src, real *dst, int st,int ed);

    EquationInfo update_no_check(int kl, int k, int kr);
    EquationInfo update_uppper_no_check(int i, int kr);
    EquationInfo update_lower_no_check(int kl, int i);

    void tpr_backward();

    void replace_with_init(int i);
    void replace_with_st1(int i);

    void tpr_stage1(int st, int ed);

    void patch_equation_info(EquationInfo eqi);
};
