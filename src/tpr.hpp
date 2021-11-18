#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <array>
#include <cmath>

#include "cr.hpp"
#include "lib.hpp"

#ifdef CR_SINGLE_THREAD
using namespace CRSingleThread;
#endif

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
#define SAFE_DELETE(p) \
    delete[] p;        \
    p = nullptr

namespace TPR_Helpers {
/**
 * @brief      Infomation of equation and its index
 */
struct EquationInfo {
    int idx;
    real a;
    real c;
    real rhs;
};
}  // namespace TPR_Helpers

class TPR : Solver {
    real *a, *c, *rhs, *x;
    real *aa, *cc, *rr;
    real *st2_a, *st2_c, *st2_rhs;
    real *bkup_a, *bkup_c, *bkup_rhs;
    CR st2solver;
    int n, s, m;

   public:
    TPR(real *a, real *diag, real *c, real *rhs, int n, int s) {
        init(n, s);
        set_tridiagonal_system(a, c, rhs);
    };

    TPR(int n, int s) { init(n, s); };

    ~TPR() {
        // free local variables
        SAFE_DELETE(this->x);
        SAFE_DELETE(this->aa);
        SAFE_DELETE(this->cc);
        SAFE_DELETE(this->rr);
        SAFE_DELETE(this->st2_a);
        SAFE_DELETE(this->st2_c);
        SAFE_DELETE(this->st2_rhs);
        SAFE_DELETE(this->bkup_a);
        SAFE_DELETE(this->bkup_c);
        SAFE_DELETE(this->bkup_rhs);
    }

    TPR(const TPR &tpr) {
        n = tpr.n;
        s = tpr.s;
        init(tpr.n, tpr.s);
    };

    void set_tridiagonal_system(real *a, real *c, real *rhs);

    void clear();

    int solve();

    int get_ans(real *x);

   private:
    void init(int n, int s);

    TPR_Helpers::EquationInfo update_no_check(int kl, int k, int kr);
    TPR_Helpers::EquationInfo update_uppper_no_check(int k, int kr);
    TPR_Helpers::EquationInfo update_lower_no_check(int kl, int k);

    void st3_replace();

    void tpr_stage1();
    void tpr_inter();
    void tpr_stage2();
    void tpr_stage3();
};
