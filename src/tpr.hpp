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
    EquationInfo *bkup_st1;
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
        SAFE_DELETE(this->bkup_st1);
        SAFE_DELETE(this->x);
    }
 
    void set_tridiagonal_system(real *a, real *c, real *rhs);

    void clear();

    int solve();

    int get_ans(real *x);

private:
    TPR(const TPR &tpr);
    TPR &operator=(const TPR &tpr);

    void init(int n, int s, pm_lib::PerfMonitor *pm);

    EquationInfo update_section(int i, int u);
    EquationInfo update_global(int i, int u);
    EquationInfo update_bd_check(int i, int u, int lb, int ub);

    void mk_bkup_st1(int st, int ed);
    void bkup_cp(int src_idx, int dst_index);

    EquationInfo update_no_check(int kl, int k, int kr);
    EquationInfo update_uppper_no_check(int k, int kr);
    EquationInfo update_lower_no_check(int kl, int k);

    void st3_replace(int st, int ed);

    void tpr_stage1(int st, int ed);
    void tpr_stage2();
    void tpr_stage3(int st, int ed);

    void patch_equation_info(EquationInfo eqi);
};
