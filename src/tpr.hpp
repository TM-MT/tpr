#pragma once
#include <assert.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <array>

#include "PerfMonitor.h"
#include "lib.hpp"
#include "pm.hpp"

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
    real *st2_a, *st2_c, *st2_rhs;
    EquationInfo *st2_use;
    int n, s;
    pm_lib::PerfMonitor *pm;
    std::array<std::string, 3> default_labels = { "st1", "st2", "st3" };
    std::array<std::string, 3> labels;

public:
    TPR(real *a, real *diag, real *c, real *rhs, int n, int s, pm_lib::PerfMonitor *pm) {
        init(n, s, pm);
        set_tridiagonal_system(a, c, rhs);
    };

    TPR(int n, int  s, pm_lib::PerfMonitor *pm) {
        init(n, s, pm);
    };

    ~TPR() {
        // free local variables
        SAFE_DELETE(this->st2_use);
        SAFE_DELETE(this->x);
        SAFE_DELETE(this->st2_a);
        SAFE_DELETE(this->st2_c);
        SAFE_DELETE(this->st2_rhs);
    }
 
    void set_tridiagonal_system(real *a, real *c, real *rhs);

    void clear();

    int solve();

    int get_ans(real *x);

private:
    TPR(const TPR &tpr);
    TPR &operator=(const TPR &tpr);

    void init(int n, int s, pm_lib::PerfMonitor *pm);

    EquationInfo update_no_check(int kl, int k, int kr);
    EquationInfo update_uppper_no_check(int k, int kr);
    EquationInfo update_lower_no_check(int kl, int k);

    void tpr_stage1(int st, int ed);
    void tpr_stage2();
    void tpr_stage3(int st, int ed);
};
