#include <assert.h>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "lib.hpp"
#include "tpr.hpp"


int TPR::solve() {
    tpr_forward();
    tpr_backward();

    return 0;
}

/**
 * @brief      TPR FORWARD REDUCTION
 */
void TPR::tpr_forward() {
    int n = this->n;
    int s = this->s;

    int k, u, p;
    
    mk_bkup_init();

    // STAGE 1
    for (k = 1; k <= fllog2(s); k += 1) {
        u = pow2(k-1);
        p = 0;

        while (p < n) {
            EquationInfo eqbuff[s];
            int j = 0;  // j <= s
    
            for (int i = p; i < p + s; i += pow2(k)) {
                eqbuff[j] = update_section(i, u);
                j += 1;
            }
            // p >= 0, k >= 1 -> i >= 0
            for (int i=p+pow2(k)-1; i < p+s; i += pow2(k)) {
                eqbuff[j] = update_section(i, u);
                j += 1;
            }

            assert(j <= s);
            for (int i = 0; i < j; i++) {
                patch_equation_info(eqbuff[i]);
            }

            p += s;
        }
    }

    // k = k + 1;
    u = pow2(k - 1);
    p = 0;
    while (p < n) {
        EquationInfo eqi = update_uppper_no_check(p, p + (u - 1));
        patch_equation_info(eqi);
        p = p + s;
    }

    mk_bkup_st1();

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

    
    // STAGE 2
    int j = 0;
    while ( j < (int)log2((double)n / s) - 1) {
        u = pow2(k-1);

        EquationInfo eqbuff[n / s];
        int idx = 0;
        for (int i = pow2(k)-1; i < n; i += pow2(k)) {
            eqbuff[idx] = update_global(i, u);
            idx += 1;
        }
        j += 1;
        k += 1;

        for (int i = 0; i < idx; i++) {
            patch_equation_info(eqbuff[i]);
        }
    }
}


void TPR::tpr_stage1(int st, int ed) {}

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
 */
void TPR::mk_bkup_init() {
    bkup_cp(this->a, this->init_a, 0, this->n);
    bkup_cp(this->diag, this->init_diag, 0, this->n);
    bkup_cp(this->c, this->init_c, 0, this->n);
    bkup_cp(this->rhs, this->init_rhs, 0, this->n);
}

/**
 * @brief      make backup equation for STAGE 3 use.
 */
void TPR::mk_bkup_st1() {
    bkup_cp(this->a, this->st1_a, 1, this->n);
    bkup_cp(this->diag, this->st1_diag, 1, this->n);
    bkup_cp(this->c, this->st1_c, 1, this->n);
    bkup_cp(this->rhs, this->st1_rhs, 1, this->n);    
}

void TPR::bkup_cp(real *src, real *dst, int st,int ed) {
    for (int i = st; i < ed; i += 2) {
        dst[i / 2] = src[i];
    }
}


/**
 * @brief      TPR BACKWARD SUBSTITUTION
 */
void TPR::tpr_backward() {
    int capital_i = n / 2;
    int m = n / s;

    int u = capital_i;
    int i = capital_i-1;

    // STAGE 2 (continue)
    {
        real inv_det = 1.0 / (diag[i+u]*diag[i] - c[i]*a[i+u]);
        
        x[i] = (diag[i+u]*rhs[i] - c[i]*rhs[i+u]) * inv_det;
        x[i+u] =  (rhs[i+u]*diag[i] - rhs[i]*a[i+u]) * inv_det;
    }    

    int j = 0;
    while (j < fllog2(m) - 1) {
        capital_i /= 2;
        u /= 2;
    
        assert(u > 0);
        real new_x[n / 2*u];
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
    
    for (i = 0; i < n; i += 1) {
        if (i % 2 == 0) {
            replace_with_init(i);
        } else {
            replace_with_st1(i);
        }
    }


    // STAGE 3
    // for (j = 0; j < fllog2(s); j += 1) {
    while (u >= 2) {
        capital_i = capital_i / 2;
        u = u / 2;
        int p = 0;

        assert(u > 0);
        assert(capital_i > 0);
        while (p < n) {
            real new_x[n / 2*u];
            int idx = 0;
            for (i = p + capital_i - 1; i < p + s; i += 2*u) {
                // update x[i]
                new_x[idx] = (rhs[i] - a[i] * x[i-u] - c[i]*x[i+u]) / diag[i];
                idx += 1;
            }

            for (i = p + capital_i - 1, idx = 0; i < p + s; i += 2*u, idx++) {
                x[i] = new_x[idx];
            }

            p += s;
        }
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
