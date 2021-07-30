#include <assert.h>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "lib.hpp"
#include "pcr.hpp"
#include "tpr.hpp"

#ifdef __GNUC__
#define UNREACHABLE __builtin_unreachable()
#else
#define UNREACHABLE
#endif


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
    // allocation for stage 2 use
    RMALLOC(this->st2_a, n / s);
    RMALLOC(this->st2_c, n / s);
    RMALLOC(this->st2_rhs, n / s);

    this->st2_use = new EquationInfo[2 * n / s];

    // NULL CHECK
    {
        bool none_null = true;
        none_null = none_null && (this->x != nullptr);
        none_null = none_null && (this->st2_a != nullptr);
        none_null = none_null && (this->st2_c != nullptr);
        none_null = none_null && (this->st2_rhs != nullptr);
        none_null = none_null && (this->st2_use != nullptr);

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
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int st = 0; st < this->n; st += s) {
            tpr_stage1(st, st + s - 1);
        }

        #pragma omp single
        {
            tpr_stage2();
        }

        #pragma omp for schedule(static)
        for (int st = 0; st < this->n; st += s) {
            tpr_stage3(st, st + s - 1);
        }
    }

    int m = n / s;
    return m * ((14 * s - 10) * fllog2(s) + 14) // stage 1
            + 28 * m - 33 // stage 2
            + m * (4 * s + 1); // stage 3
}


/**
 * @brief      TPR STAGE 1
 *
 * @param[in]  st     start index of equation that this function calculate
 * @param[in]  ed     end index of equation that this function calculate
 */
void TPR::tpr_stage1(int st, int ed) {
    // local copies of a, c, rhs
    real loc_a[s], loc_c[s], loc_rhs[s];

    // make local copies
    for (int i = st; i <= ed; i++) {
        loc_a[i - st] = this->a[i];
        loc_c[i - st] = this->c[i];
        loc_rhs[i - st] = this->rhs[i];
    }

    for (int k = 1; k <= fllog2(s); k += 1) {
        const int u = pow2(k-1);
        const int s = this->s;

        // Temp arrays for a, c, rhs
        real aa[s], cc[s], rr[s];

        #pragma omp simd
        for (int i = 0; i < u; i++) {
            assert(i + u < s);

            // from update_uppper_no_check(i, i + u);
            int k = i;
            int kr = i + u;

            real inv_diag_k = 1.0 / (1.0 - loc_a[kr] * loc_c[k]);

            aa[i] = inv_diag_k * loc_a[k];
            cc[i] = -inv_diag_k * loc_c[kr] * loc_c[k];
            rr[i] = inv_diag_k * (loc_rhs[k] - loc_rhs[kr] * loc_c[k]);
        }

        #pragma omp simd
        for (int i = u; i < s - u; i++) {
            assert(0 <= i - u);
            assert(i + u < s);

            // from update_no_check(i - u , i, i + u);
            int kl = i - u;
            int k = i;
            int kr = i + u;
            real inv_diag_k = 1.0 / (1.0 - loc_c[kl] * loc_a[k] - loc_a[kr] * loc_c[k]);

            aa[i] = - inv_diag_k * loc_a[kl] * loc_a[k];
            cc[i] = - inv_diag_k * loc_c[kr] * loc_c[k];
            rr[i] = inv_diag_k * (loc_rhs[k] - loc_rhs[kl] * loc_a[k] - loc_rhs[kr] * loc_c[k]);
        }

        #pragma omp simd
        for (int i = s - u; i < s; i++) {
            assert(0 <= i - u);
            
            // from update_lower_no_check(i - u, i);
            int kl = i - u;
            int k = i;
            real inv_diag_k = 1.0 / (1.0 - loc_c[kl] * loc_a[k]);

            aa[i] = -inv_diag_k * loc_a[kl] * loc_a[k];
            cc[i] = inv_diag_k * loc_c[k];
            rr[i] = inv_diag_k * (loc_rhs[k] - loc_rhs[kl] * loc_a[k]);
        }

        // patch
        for (int i = 0; i < s; i++) {
            loc_a[i] = aa[i];
            loc_c[i] = cc[i];
            loc_rhs[i] = rr[i];
        }
    }

    // copy back
    for (int i = st; i <= ed; i++) {
        this->a[i] = loc_a[i - st];
        this->c[i] = loc_c[i - st];
        this->rhs[i] = loc_rhs[i - st];
    }

}

/**
 * @brief TPR STAGE 2
 *
 */
void TPR::tpr_stage2() {
    // Update by E_{st} and E_{ed} copy E_{ed} for stage 2 use
    for (int st = 0; st < this->n; st += s) {
        // EquationInfo eqi = update_uppper_no_check(st, ed);
        int k = st, kr = st + s - 1;
        real ak = a[k];
        real akr = a[kr];
        real ck = c[k];
        real ckr = c[kr];
        real rhsk = rhs[k];
        real rhskr = rhs[kr];

        real inv_diag_k = 1.0 / (1.0 - akr * ck);

        EquationInfo eqi;
        eqi.idx = st;
        eqi.a = inv_diag_k * ak;
        eqi.c = -inv_diag_k * ckr * ck;
        eqi.rhs = inv_diag_k * (rhsk - rhskr * ck);
        int eqi_dst = 2 * st / s;
        this->st2_use[eqi_dst] = eqi;

        EquationInfo eqi2;
        eqi2.idx = kr;
        eqi2.a = akr; // a.k.a. a[ed]
        eqi2.c = ckr;
        eqi2.rhs = rhskr;
        this->st2_use[eqi_dst + 1] = eqi2;
    }

    // INTERMIDIATE STAGE
    {
        int j = 0;
        int len_st2_use = 2 * n / s;
        #pragma omp simd
        for (int i = 1; i < len_st2_use - 1; i += 2) {
            int k = i;
            int kr = i + 1;
            real ak = this->st2_use[k].a;
            real akr = this->st2_use[kr].a;
            real ck = this->st2_use[k].c;
            real ckr = this->st2_use[kr].c;
            real rhsk = this->st2_use[k].rhs;
            real rhskr = this->st2_use[kr].rhs;

            real inv_diag_k = 1.0 / (1.0 - akr * ck);

            this->st2_a[j] = inv_diag_k * ak;
            this->st2_c[j] = -inv_diag_k * ckr * ck;
            this->st2_rhs[j] = inv_diag_k * (rhsk - rhskr * ck);
            j++;
        }
        this->st2_a[j] = this->st2_use[len_st2_use - 1].a;
        this->st2_c[j] = this->st2_use[len_st2_use - 1].c;
        this->st2_rhs[j] = this->st2_use[len_st2_use - 1].rhs;
    }


    PCR p = PCR(this->st2_a, nullptr, this->st2_c, this->st2_rhs, n / s);
    p.solve();
    // assert this->st2_rhs has the answer
    // copy back to TPR::x
    {
        int j = 0;
        for (int i = s - 1; i < n - s; i += s) {
           this->x[i] = this->st2_rhs[j];
           j++;
        }
        this->x[n - 1] = this->st2_rhs[j];
    }
}

/**
 * @brief      TPR STAGE 3
 *
 * @param[in]  st     start index of equation that this function calculate
 * @param[in]  ed     end index of equation that this function calculate
 */
void TPR::tpr_stage3(int st, int ed) {
    int lbi = st - 1; // use as index of the slice top
    real xl = 0.0;
    if (lbi < 0) {
        xl = 0.0; // x[-1] does not exists
    } else {
        xl = x[lbi];
    }

    real key = 0.0;
    if (ed == this->n - 1) { // c[n - 1] should be 0.0
        key = 0.0;
    } else {
        key = 1.0 / c[ed] * (rhs[ed] - a[ed] * xl - x[ed]);
    }

    // x[ed] is known
    #pragma omp simd
    for (int i = st; i < ed; i++) {
        x[i] = rhs[i] - a[i] * xl - c[i] * key;
    }
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


#undef UNREACHABLE
