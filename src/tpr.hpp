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
#define RMALLOC(x, n) x = new real[n]

/**
 * @brief Safely delete pointer `p`
 */
#define SAFE_DELETE( p ) delete p; p = nullptr


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
    real *bkup_a, *bkup_c, *bkup_rhs;
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
        RMALLOC(this->bkup_a, n);
        RMALLOC(this->bkup_c, n);
        RMALLOC(this->bkup_rhs, n);

        // NULL CHECK
        real **ps[] = { &this->bkup_a, &this->bkup_c, &this->bkup_rhs,
                        &this->x,
        };
        for (int i = 0; static_cast<long unsigned int>(i) < sizeof(ps) / sizeof(ps[0]); i++) {
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

    ~TPR() {
        // free local variables
        SAFE_DELETE(this->init_a);
        SAFE_DELETE(this->init_c);
        SAFE_DELETE(this->init_rhs);
        SAFE_DELETE(this->st1_a);
        SAFE_DELETE(this->bkup_a);
        SAFE_DELETE(this->bkup_c);
        SAFE_DELETE(this->bkup_rhs);
        SAFE_DELETE(this->x);
    }
 
    int solve();

    int get_ans(real *x);

private:
    TPR(const TPR &tpr);
    TPR &operator=(const TPR &tpr);

    EquationInfo update_section(int i, int u);
    EquationInfo update_global(int i, int u);
    EquationInfo update_bd_check(int i, int u, int lb, int ub);

    void mk_bkup_init(int st, int ed);
    void mk_bkup_st1(int st, int ed);
    void bkup_cp(real *src, real *dst, int st,int ed);

    EquationInfo update_no_check(int kl, int k, int kr);
    EquationInfo update_uppper_no_check(int k, int kr);
    EquationInfo update_lower_no_check(int kl, int k);

    void st3_replace();

    void tpr_stage1(int st, int ed);
    void tpr_stage2();
    void tpr_stage3(int st, int ed);

    void patch_equation_info(EquationInfo eqi);
};
