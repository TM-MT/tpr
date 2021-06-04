#include <assert.h>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "lib.hpp"
#include "tpr.hpp"


int TPR::solve() {
    #pragma omp parallel for
    for (int st = 0; st < this->n; st += s) {
        tpr_stage1(st, st + s - 1);
    }

    tpr_stage2();

    #pragma omp parallel for
    for (int st = 0; st < this->n; st += s) {
        tpr_stage3(st, st + s - 1);
    }
    return 0;
}


/**
 * @brief      TPR STAGE 1
 *
 * @param[in]  st     start index of equation that this function calculate
 * @param[in]  ed     end index of equation that this function calculate
 */
void TPR::tpr_stage1(int st, int ed) {
    mk_bkup_init(st, ed);

    for (int k = 1; k <= fllog2(s); k += 1) {
        int u = pow2(k-1);

        EquationInfo eqbuff[s];
        int j = 0;  // j <= s

        for (int i = st; i <= ed; i += pow2(k)) {
            eqbuff[j] = update_section(i, u);
            j += 1;
        }
        // ed >= 0, k >= 1 -> i >= 0
        for (int i=st+pow2(k)-1; i <= ed; i += pow2(k)) {
            eqbuff[j] = update_section(i, u);
            j += 1;
        }

        assert(j <= s);
        for (int i = 0; i < j; i++) {
            patch_equation_info(eqbuff[i]);
        }
    }

    EquationInfo eqi = update_uppper_no_check(st, ed);
    patch_equation_info(eqi);

    mk_bkup_st1(st, ed);
}

/**
 * @brief TPR STAGE 2
 *
 */
void TPR::tpr_stage2() {
    // INTERMIDIATE STAGE
    {
        EquationInfo eqbuff[n / s];
        int j = 0;
        // for (int i = s; i <= n - s; i += s) {
        for (int i = s-1; i < n - s; i += s) {
            eqbuff[j] = update_uppper_no_check(i, i + 1);
            j += 1;
        }

        assert(j <= n / s);
        for (int i = 0; i < j; i++) {
            patch_equation_info(eqbuff[i]);
        }
    }


    // STAGE 2 FR (CR FORWARD REDUCTION)
    {

        // int j = 0;
        // while ( j < (int)log2((double)n / s) - 1) {
        for (int j = 0, k = log2(s); j < (int)log2((double)n / s) - 1; j++, k++) {
            int u = pow2(k-1);

            EquationInfo eqbuff[n / s];
            int idx = 0;
            for (int i = pow2(k)-1; i < n; i += pow2(k)) {
                eqbuff[idx] = update_global(i, u);
                idx += 1;
            }
            // j += 1;
            // k += 1;

            for (int i = 0; i < idx; i++) {
                patch_equation_info(eqbuff[i]);
            }
        }
    }


    int capital_i = n / 2;
    int m = n / s;

    int u = capital_i;
    int i = capital_i-1;

    // CR BACKWARD SUBSTITUTION STEP 1
    {
        real inv_det = 1.0 / (diag[i+u]*diag[i] - c[i]*a[i+u]);

        x[i] = (diag[i+u]*rhs[i] - c[i]*rhs[i+u]) * inv_det;
        x[i+u] =  (rhs[i+u]*diag[i] - rhs[i]*a[i+u]) * inv_det;
    }

    // CR BACKWARD SUBSTITUTION
    {
        int j = 0;
        while (j < fllog2(m) - 1) {
            capital_i /= 2;
            u /= 2;

            assert(u > 0);
            real new_x[n / (2*u)];
            int idx = 0;
            for (i = capital_i - 1; i < n; i += 2*u) {
                new_x[idx] = (rhs[i] - a[i]*x[i-u] - c[i]*x[i+u]) / diag[i];
                idx += 1;
            }

            idx = 0;
            for (i = capital_i - 1; i < n; i += 2*u) {
                x[i] = new_x[idx];
                idx += 1;
            }

            j += 1;
        }
    }
}

/**
 * @brief      TPR STAGE 3
 *
 * @param[in]  st     start index of equation that this function calculate
 * @param[in]  ed     end index of equation that this function calculate
 */
void TPR::tpr_stage3(int st, int ed) {
    // REPLACING
    for (int i = st; i <= ed; i += 1) {
        if (i % 2 == 0) {
            replace_with_init(i);
        } else {
            replace_with_st1(i);
        }
    }

    int capital_i = s;
    int u = s;
    for (int j = 0; j < fllog2(s); j += 1) {
    // while (u >= 2) {
        capital_i = capital_i / 2;
        u = u / 2;

        assert(u > 0);
        assert(capital_i > 0);
        real new_x[n / (2*u)];
        int idx = 0;
        for (int i = st + capital_i - 1; i <= ed; i += 2*u) {
            // update x[i]
            new_x[idx] = (rhs[i] - a[i] * x[i-u] - c[i]*x[i+u]) / diag[i];
            idx += 1;
        }

        assert(idx <= n / (2*u));
        for (int i = st + capital_i - 1, idx = 0; i <= ed; i += 2*u, idx++) {
            x[i] = new_x[idx];
        }
    }
}

/**
 * @brief      reduction calculation at section
 *
 * @param[in]  i     index of equation 
 * @param[in]  u     index of equation to use
 *
 * @return     The equation information which was performed reduction.
 */
EquationInfo TPR::update_section(int i, int u) {    
    int lb = i / s * s;
    int ub = lb + s;
    
    return update_bd_check(i, u, lb, ub);
}

/**
 * @brief      reduction calculation at STAGE 2
 *
 * @param[in]  i     index of equation 
 * @param[in]  u     index of equation to use
 *
 * @return     The equation information which was performed reduction.
 */
EquationInfo TPR::update_global(int i, int u) {
    return update_bd_check(i, u, 0, n);
}


/**
 * @brief      check boundary condition $i-u, i+u \in [lb, ub)$ and call update_*()
 *
 * @param[in]  i     index of equation 
 * @param[in]  u     index of equation to use
 * @param[in]  lb    The lower bound
 * @param[in]  ub    The uppper bound
 *
 * @return     The equation information which was performed reduction.
 */
EquationInfo TPR::update_bd_check(int i, int u, int lb, int ub) {
    bool lb_check = lb <= i - u;
    bool ub_check = i + u < ub;
    EquationInfo eqi;

    if (lb_check && ub_check) {
        eqi = update_no_check(i - u, i, i + u);    
    } else if (ub_check) {
        eqi = update_uppper_no_check(i, i + u);
    } else if (lb_check) {
        eqi = update_lower_no_check(i - u, i);
    } else {
        // not happen
        assert(false);
    }

    return eqi;
}

/// Update E_i by E_{kl}, E_{kr}
EquationInfo TPR::update_no_check(int kl, int k, int kr) {
    assert(0 <= kl && kl < k && k < kr && kr < n);
    real ai = a[k];
    real diagi = diag[k];
    real ci = c[k];
    real rhsi = rhs[k];

    real s1 = ai / diag[kl];
    real s2 = ci / diag[kr];

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = - a[kl] * s1;
    eqi.diag = diagi - c[kl] * s1 - a[kr] * s2;
    eqi.c = - c[kr] * s2;
    eqi.rhs = rhsi - rhs[kl] * s1 - rhs[kr] * s2;

    return eqi;
}


/// Update E_i by E_{kr}
EquationInfo TPR::update_uppper_no_check(int i, int kr) {
    assert(0 <= i && i < kr && kr < n);

    real s2 = c[i] / diag[kr];

    EquationInfo eqi;
    eqi.idx = i;
    eqi.a = a[i];  // no update for a[i]
    eqi.diag = diag[i] - a[kr] * s2;
    eqi.c = -c[kr] * s2;
    eqi.rhs = rhs[i] - rhs[kr] * s2;

    return eqi;
}

/// Update E_i by E_{kl}
EquationInfo TPR::update_lower_no_check(int kl, int i) {
    assert(0 <= kl && kl < i && i < n);

    real s1 = a[i] / diag[kl];

    EquationInfo eqi;
    eqi.idx = i;
    eqi.a = - a[kl] * s1;
    eqi.diag = diag[i] - c[kl] * s1;
    eqi.c = c[i];  // no update for c[i]
    eqi.rhs = rhs[i] - rhs[kl] * s1;

    return eqi;
}


/**
 * @brief      make backup equation for STAGE 3 use.
 *
 * @param[in]    st    start index of equation
 * @param[in]    ed    end index of equation
 */
void TPR::mk_bkup_init(int st, int ed) {
    int stidx = st;
    bkup_cp(this->a, this->init_a, stidx, ed);
    bkup_cp(this->diag, this->init_diag, stidx, ed);
    bkup_cp(this->c, this->init_c, stidx, ed);
    bkup_cp(this->rhs, this->init_rhs, stidx, ed);
}

/**
 * @brief      make backup equation for STAGE 3 use.
 *
 * @param[in]    st    start index of equation
 * @param[in]    ed    end index of equation
 */
void TPR::mk_bkup_st1(int st, int ed) {
    int stidx = st + 1;
    bkup_cp(this->a, this->st1_a, stidx, ed);
    bkup_cp(this->diag, this->st1_diag, stidx, ed);
    bkup_cp(this->c, this->st1_c, stidx, ed);
    bkup_cp(this->rhs, this->st1_rhs, stidx, ed);
}

void TPR::bkup_cp(real *src, real *dst, int st,int ed) {
    for (int i = st; i < ed; i += 2) {
        dst[i / 2] = src[i];
    }
}


/**
 * @brief      replace the E_i by INITIAL STATE
 * 
 * call `TPR::mk_bkup_init` before use to make sure `this->init_*` has correct information.
 *
 * @param[in]  i     index to replace
 */
void TPR::replace_with_init(int i) {
    int bkup_idx = i / 2;
    this->a[i] = this->init_a[bkup_idx];
    this->diag[i] = this->init_diag[bkup_idx];
    this->c[i] = this->init_c[bkup_idx];
    this->rhs[i] = this->init_rhs[bkup_idx];
}

/**
 * @brief      replace the E_i by STAGE 1 STATE
 * 
 * call `TPR::mk_bkup_st1` before use to make sure `this->st1_*` has correct information.
 *
 * @param[in]  i     index to replace
 */
void TPR::replace_with_st1(int i) {
    int bkup_idx = i / 2;
    this->a[i] = this->st1_a[bkup_idx];
    this->diag[i] = this->st1_diag[bkup_idx];
    this->c[i] = this->st1_c[bkup_idx];
    this->rhs[i] = this->st1_rhs[bkup_idx];
}


int TPR::get_ans(real *x) {
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};

/**
 * @brief      patch equation by given equation information
 *
 * @param[in]  eqi   The eqi
 */
void TPR::patch_equation_info(EquationInfo eqi) {
    int idx = eqi.idx;
    this->a[idx] = eqi.a;
    this->diag[idx] = eqi.diag;
    this->c[idx] = eqi.c;
    this->rhs[idx] = eqi.rhs;
}
