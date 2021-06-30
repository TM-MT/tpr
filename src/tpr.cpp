#include <assert.h>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "lib.hpp"
#include "tpr.hpp"


/**
 * @brief set Tridiagonal System for TPR
 *
 * USAGE
 * ```
 * TPR t(n, s);
 * t.set_tridiagonal_system(a, c, rhs);
 * t.solve();
 * t.get_ans(x);
 * ```
 *
 * @param a
 * @param c
 * @param rhs
 */
void TPR::set_tridiagonal_system(real *a, real *c, real *rhs) {
    this->a = a;
    this->c = c;
    this->rhs = rhs;
}

/**
 * @brief Initializer for TPR()
 * @details Call this function first to set up for TPR
 *
 * @note This function should call once.
 *
 * @param n size of given system
 * @param s size of a slice. `s` should be power of 2
 */
void TPR::init(int n, int s) {
    this->n = n;
    this->s = s;
    // allocation for answer
    RMALLOC(this->x, n);

    this->bkup_st1 = new EquationInfo[2 * n / s];

    // NULL CHECK
    {
        bool none_null = true;
        none_null = none_null && (this->x != nullptr);
        none_null = none_null && (this->bkup_st1 != nullptr);

        if (!none_null) {
            printf("[%s] FAILED TO ALLOCATE an array.\n",
                __func__
                );
            abort();
        }
    }
}


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
    for (int k = 1; k <= fllog2(s); k += 1) {
        int u = pow2(k-1);

        EquationInfo eqbuff[s];
        int j = 0;  // j <= s

        for (int i = st; i <= ed; i++, j++) {
            eqbuff[j] = update_section(i, u);
        }

        assert(j == s);  // ed - st + 1 == s
        for (int i = 0; i < j; i++) {
            patch_equation_info(eqbuff[i]);
        }
    }

    // make backup for STAGE 3 use
    mk_bkup_st1(st, ed);

    EquationInfo eqi = update_uppper_no_check(st, ed);
    patch_equation_info(eqi);
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
    // replace
    st3_replace(st, ed);

    int lbi = st - 1; // use as index of the slice top
    real xl;
    if (lbi < 0) {
        xl = 0.0; // x[-1] does not exists
    } else {
        xl = x[lbi];
    }

    real key;
    if (c[ed] == 0.0) { // c[n] should be 0.0
        key = 0.0;
    } else {
        key = 1.0 / c[ed] * (rhs[ed] - a[ed] * xl - x[ed]);
    }

    // x[ed] is known
    for (int i = st; i < ed; i++) {
        x[i] = rhs[i] - a[i] * xl - c[i] * key;
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
    real akl = a[kl];
    real ck = c[k];
    real ckl = c[kl];
    real rhskl = rhs[kl];
    real rhsk = rhs[k];

    real inv_diag_k = 1.0 / (1.0 - ckl * ak);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = -inv_diag_k * akl * ak;
    eqi.c = inv_diag_k * ck;
    eqi.rhs = inv_diag_k * (rhsk - rhskl * ak);

    return eqi;
}


/**
 * @brief      make backup equation for STAGE 3 use.
 *
 * @param[in]    st    E_{st}
 * @param[in]    ed    E_{ed}
 */
void TPR::mk_bkup_st1(int st, int ed) {
    int eqi_st = 2 * st / s;

    bkup_cp(st, eqi_st);
    bkup_cp(ed, eqi_st + 1);
}


/**
 * @brief      copy Equation_{src_idx} to the backup array EquationInfo[dst_index]
 *
 * @param[in]  src_idx    The source index
 * @param[in]  dst_index  The destination index of EquationInfo
 */
void TPR::bkup_cp(int src_idx, int dst_index) {
    assert(0 <= dst_index && dst_index < 2 * n / s);

    this->bkup_st1[dst_index].idx = src_idx;
    this->bkup_st1[dst_index].a = this->a[src_idx];
    this->bkup_st1[dst_index].c = this->c[src_idx];
    this->bkup_st1[dst_index].rhs = this->rhs[src_idx];
}

/**
 * @brief   subroutine for STAGE 3 REPLACE
 * 
 * @param[in]   st    
 * @param[in]   ed
 * 
 * @note    make sure `bkup_st1` was allocated and mk_bkup_st1 functions was called.
 */
void TPR::st3_replace(int st, int ed) {
    int eqi_st = st / s * 2;

    patch_equation_info(this->bkup_st1[eqi_st]);
    patch_equation_info(this->bkup_st1[eqi_st + 1]);
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
