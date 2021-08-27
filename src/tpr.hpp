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
 * @brief Safely delete pointer `p` and set `p = nullptr`
 */
#define SAFE_DELETE( p ) delete[] p; p = nullptr


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
    real *aa, *cc, *rr;
    real *st2_a, *st2_c, *st2_rhs;
    real *inter_a, *inter_c, *inter_rhs;
    real *bkup_a, *bkup_c, *bkup_rhs;
    int n, s, m;

public:
    TPR(real *a, real *diag, real *c, real *rhs, int n, int s) {
        init(n, s);
        set_tridiagonal_system(a, c, rhs);
    };

    TPR(int n, int  s) {
        init(n, s);
    };

    ~TPR() {
        // free local variables
        SAFE_DELETE(this->x);
        SAFE_DELETE(this->aa);
        SAFE_DELETE(this->cc);
        SAFE_DELETE(this->rr);
        SAFE_DELETE(this->st2_a);
        SAFE_DELETE(this->st2_c);
        SAFE_DELETE(this->st2_rhs);
        SAFE_DELETE(this->inter_a);
        SAFE_DELETE(this->inter_c);
        SAFE_DELETE(this->inter_rhs);
        SAFE_DELETE(this->bkup_a);
        SAFE_DELETE(this->bkup_c);
        SAFE_DELETE(this->bkup_rhs);
        #ifdef _OPENACC
        #pragma acc exit data delete(aa[:n], cc[:n], rr[:n])
        #pragma acc exit data delete(this->x[:n])
        #pragma acc exit data delete(this->st2_a[:n/s], this->st2_c[:n/s], this->st2_rhs[:n/s])
        #pragma acc exit data delete(this->inter_a[:2*n/s], this->inter_c[:2*n/s], this->inter_rhs[:2*n/s])
        #pragma acc exit data delete(this->n, this->s, this)
        #endif
    }

    void set_tridiagonal_system(real *a, real *c, real *rhs);

    void clear();

    int solve();

    int get_ans(real *x);

private:
    TPR(const TPR &tpr);
    TPR &operator=(const TPR &tpr);

    void init(int n, int s);
    #pragma acc routine vector
    void mk_bkup_init(int st, int ed);
    #pragma acc routine vector
    void mk_bkup_st1(int st, int ed);
    #pragma acc routine vector
    void bkup_cp(real *src, real *dst, int st,int ed);

    EquationInfo update_no_check(int kl, int k, int kr);
    EquationInfo update_uppper_no_check(int k, int kr);
    EquationInfo update_lower_no_check(int kl, int k);

    void st3_replace();

    void tpr_stage1(int st, int ed);
    void tpr_stage2();
    #pragma acc routine gang
    void tpr_stage3(int st, int ed);
};
