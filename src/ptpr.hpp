#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <array>
#include <cmath>

#include "lib.hpp"
#include "pcr.hpp"

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

namespace PTPR_Helpers {
/**
 * @brief      Infomation of equation and its index
 */
struct EquationInfo {
    int idx;
    real a;
    real c;
    real rhs;
};
}  // namespace PTPR_Helpers

class PTPR : Solver {
    real *a, *c, *rhs, *x;
    real *aa, *cc, *rr;
    real *st2_a, *st2_c, *st2_rhs;
    real *inter_a, *inter_c, *inter_rhs;
    PCR st2solver;
    int n, s, m;

   public:
    PTPR(real *a, real *diag, real *c, real *rhs, int n, int s) {
        init(n, s);
        set_tridiagonal_system(a, c, rhs);
    };

    PTPR(int n, int s) { init(n, s); };

    ~PTPR() {
        // free local variables
        delete[] & this->x[-1];
        SAFE_DELETE(this->aa);
        SAFE_DELETE(this->cc);
        SAFE_DELETE(this->rr);
        SAFE_DELETE(this->st2_a);
        SAFE_DELETE(this->st2_c);
        SAFE_DELETE(this->st2_rhs);
        SAFE_DELETE(this->inter_a);
        SAFE_DELETE(this->inter_c);
        SAFE_DELETE(this->inter_rhs);
    }

    PTPR(const PTPR &ptpr) {
        n = ptpr.n;
        s = ptpr.s;
        init(ptpr.n, ptpr.s);
    };

    void set_tridiagonal_system(real *a, real *c, real *rhs);

    void clear();

    int solve();

    int get_ans(real *x);

   private:
    void init(int n, int s);

    PTPR_Helpers::EquationInfo update_no_check(int kl, int k, int kr);
    PTPR_Helpers::EquationInfo update_uppper_no_check(int k, int kr);
    PTPR_Helpers::EquationInfo update_lower_no_check(int kl, int k);

    void tpr_stage1();
    void tpr_stage2();
    void tpr_stage3();
};
