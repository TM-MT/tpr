#include <assert.h>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "lib.hpp"
#include "tpr.hpp"

/**
 * @brief solve
 * @return num of float operation
 */
int TPR::solve() {
    #pragma omp parallel for
    for (int st = 0; st < this->n; st += s) {
        tpr_stage1(st, st + s - 1);
    }

    tpr_stage2();

    st3_replace();

    #pragma omp parallel for
    for (int st = 0; st < this->n; st += s) {
        tpr_stage3(st, st + s - 1);
    }

    int m = n / s;
    return 24 * m * (s - 2) + 60 * m - 36 + 5 * n;
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
        for (int j = 0, k = fllog2(s); j < fllog2(n / s) - 1; j++, k++) {
            int u = pow2(k-1);

            EquationInfo eqbuff[n / s];
            int idx = 0;
            for (int i = pow2(k)-1; i < n; i += pow2(k)) {
                eqbuff[idx] = update_global(i, u);
                idx += 1;
            }

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
        real inv_det = 1.0 / (1.0 - c[i]*a[i+u]);

        x[i] = (rhs[i] - c[i]*rhs[i+u]) * inv_det;
        x[i+u] =  (rhs[i+u] - rhs[i]*a[i+u]) * inv_det;
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
                new_x[idx] = rhs[i] - a[i]*x[i-u] - c[i]*x[i+u];
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
    // REPLACE should have done.

    int capital_i = s;
    int u = s;
    for (int j = 0; j < fllog2(s); j += 1) {
        capital_i = capital_i / 2;
        u = u / 2;

        assert(u > 0);
        assert(capital_i > 0);
        real new_x[n / (2*u)];
        int idx = 0;
        for (int i = st + capital_i - 1; i <= ed; i += 2*u) {
            // update x[i]
            new_x[idx] = rhs[i] - a[i] * x[i-u] - c[i]*x[i+u];
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
 * Update E_i by using E_{i-u} and E_{i+u}. Do boundary check to make sure operation held in the section.
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
 * Update E_i by using E_{i-u} and E_{i+u}. Do boundary check to avoid segmentation fault.
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

/// Update E_k by E_{kl}, E_{kr}
EquationInfo TPR::update_no_check(int kl, int k, int kr) {
    assert(0 <= kl && kl < k && k < kr && kr < n);
    real akl = a[kl];
    real ak = a[k];
    real akr = a[kr];
    real ckl = c[kl];
    real ck = c[k];
    real ckr = c[kr];
    real rhskl = rhs[kl];
    real rhsk = rhs[k];
    real rhskr = rhs[kr];

    real inv_diag_k = 1.0 / (1.0 - ckl * ak - akr * ck);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = - inv_diag_k * akl * ak;
    eqi.c = - inv_diag_k * ckr * ck;
    eqi.rhs = inv_diag_k * (rhsk - rhskl * ak - rhskr * ck);

    return eqi;
}


/// Update E_k by E_{kr}
EquationInfo TPR::update_uppper_no_check(int k, int kr) {
    assert(0 <= k && k < kr && kr < n);
    real ak = a[k];
    real akr = a[kr];
    real ck = c[k];
    real ckr = c[kr];
    real rhsk = rhs[k];
    real rhskr = rhs[kr];

    real inv_diag_k = 1.0 / (1.0 - akr * ck);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = inv_diag_k * ak;
    eqi.c = -inv_diag_k * ckr * ck;
    eqi.rhs = inv_diag_k * (rhsk - rhskr * ck);

    return eqi;
}

/// Update E_k by E_{kl}
EquationInfo TPR::update_lower_no_check(int kl, int k) {
    assert(0 <= kl && kl < k && k < n);
    real ak = a[k];
    real ck = c[k];
    real rhskl = rhs[kl];
    real rhsk = rhs[k];

    real inv_diag_k = 1.0 / (1.0 - c[kl] * ak);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = -inv_diag_k * a[kl] * ak;
    eqi.c = inv_diag_k * ck;
    eqi.rhs = inv_diag_k * (rhsk - rhskl * ak);

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
    bkup_cp(this->a, this->bkup_a, stidx, ed);
    bkup_cp(this->c, this->bkup_c, stidx, ed);
    bkup_cp(this->rhs, this->bkup_rhs, stidx, ed);
}

/**
 * @brief      make backup equation for STAGE 3 use.
 *
 * @param[in]    st    start index of equation
 * @param[in]    ed    end index of equation
 */
void TPR::mk_bkup_st1(int st, int ed) {
    int stidx = st + 1;
    bkup_cp(this->a, this->bkup_a, stidx, ed);
    bkup_cp(this->c, this->bkup_c, stidx, ed);
    bkup_cp(this->rhs, this->bkup_rhs, stidx, ed);
}

void TPR::bkup_cp(real *src, real *dst, int st,int ed) {
    for (int i = st; i <= ed; i += 2) {
        dst[i] = src[i];
    }
}

/**
 * @brief   subroutine for STAGE 3 REPLACE
 *
 * @note    make sure `bkup_*` are allocated and call mk_bkup_* functions
 */
void TPR::st3_replace() {
    std::swap(this->a, this->bkup_a);
    std::swap(this->c, this->bkup_c);
    std::swap(this->rhs, this->bkup_rhs);
}

/**
 * @brief get the answer
 * @return num of float operation
 */
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
    this->c[idx] = eqi.c;
    this->rhs[idx] = eqi.rhs;
}
