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
    real *bkup_a, *bkup_c, *bkup_rhs;
    PCR st2solver;
    int n, s, m, sl;

   public:
    PTPR(real *a, real *diag, real *c, real *rhs, int n, int s) {
        init(n, s);
        set_tridiagonal_system(a, c, rhs);
    };

    PTPR(int n, int s) { init(n, s); };

    ~PTPR() {
        // extend_input_array created local variable
        SAFE_DELETE(this->a);
        SAFE_DELETE(this->c);
        SAFE_DELETE(this->rhs);
        // free local variables
        delete[] & this->x[-1];
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
    void tpr_stage1();
    void tpr_inter();
    void tpr_stage2();
    void tpr_stage3();

    real *extend_input_array(real *p);
};
