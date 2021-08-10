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
    #pragma acc enter data copyin(a[:n], c[:n], rhs[:n])
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
    #pragma acc enter data copyin(this, this->n, this->s)
    // allocation for answer
    RMALLOC(this->x, n);
    // allocation for stage 1 use
    RMALLOC(this->aa, n);
    RMALLOC(this->cc, n);
    RMALLOC(this->rr, n);
    // allocation for stage 2 use
    RMALLOC(this->st2_a, n / s);
    RMALLOC(this->st2_c, n / s);
    RMALLOC(this->st2_rhs, n / s);

    RMALLOC(this->inter_a, 2 * n / s);
    RMALLOC(this->inter_c, 2 * n / s);
    RMALLOC(this->inter_rhs, 2 * n / s);
    #pragma acc enter data create(aa[:n], cc[:n], rr[:n])
    #pragma acc enter data create(x[:n])
    #pragma acc enter data create(st2_a[:n/s], st2_c[:n/s], st2_rhs[:n/s])
    #pragma acc enter data create(inter_a[:2*n/s], inter_c[:2*n/s], inter_rhs[:2*n/s])

    // NULL CHECK
    {
        bool none_null = true;
        none_null = none_null && (this->x != nullptr);
        none_null = none_null && (this->st2_a != nullptr);
        none_null = none_null && (this->st2_c != nullptr);
        none_null = none_null && (this->st2_rhs != nullptr);
        none_null = none_null && (this->inter_a != nullptr);
        none_null = none_null && (this->inter_c != nullptr);
        none_null = none_null && (this->inter_rhs != nullptr);

        if (!none_null) {
            printf("[%s] FAILED TO ALLOCATE an array.\n",
                __func__
                );
            abort();
        }
    }
}


/**
 * @brief      TPR STAGE 1
 *
 * @param[in]  st     start index of equation that this function calculate
 * @param[in]  ed     end index of equation that this function calculate
 */
void TPR::tpr_stage1(int st, int ed) {
}

/**
 * @brief solve
 * @return num of float operation
 */
int TPR::solve() {
    for (int k = 1; k <= static_cast<int>(log2(s)); k += 1) {
        const int u = pow2(k-1);
        const int s = this->s;

        #ifdef _OPENACC
        #pragma acc kernels present(this, a[:n], c[:n], rhs[:n], aa[:s], cc[:s], rr[:s])
        #pragma acc loop independent
        #endif
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int st = 0; st < this->n; st += s) {
            // tpr_stage1(st, st + s - 1);
            int ed = st + s - 1;


            #ifdef _OPENMP
            #pragma omp simd
            #endif
            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            for (int i = st; i < st + u; i++) {
                assert(i + u <= ed);

                // from update_uppper_no_check(i, i + u);
                int k = i;
                int kr = i + u;

                real inv_diag_k = 1.0 / (1.0 - a[kr] * c[k]);

                aa[k] = inv_diag_k * a[k];
                cc[k] = -inv_diag_k * c[kr] * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kr] * c[k]);
            }

            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int i = st + u; i <= ed - u; i++) {
                assert(st <= i - u);
                assert(i + u <= ed);

                // from update_no_check(i - u , i, i + u);
                int kl = i - u;
                int k = i;
                int kr = i + u;
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                aa[k] = - inv_diag_k * a[kl] * a[k];
                cc[k] = - inv_diag_k * c[kr] * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
            }

            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int i = ed - u + 1; i <= ed; i++) {
                assert(st <= i - u);

                // form update_lower_no_check(i - u, i);
                int kl = i - u;
                int k = i;
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k]);

                aa[k] = -inv_diag_k * a[kl] * a[k];
                cc[k] = inv_diag_k * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k]);
            }

            // patch
            #ifdef _OPENACC
            #pragma acc loop
            #endif
            for (int i = st; i <= ed; i++) {
                this->a[i] = aa[i];
                this->c[i] = cc[i];
                this->rhs[i] = rr[i];
            }
        }

    }

    #ifdef _OPENMP
    #pragma omp simd
    #endif
    tpr_stage2();

    #ifdef _OPENACC
    #pragma acc parallel present(this, a[:n], c[:n], rhs[:n], x[:n])
    #pragma acc loop gang
    #endif
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int st = 0; st < this->n; st += s) {
        // from tpr_stage3(st, st + s - 1);
        int ed = st + s - 1;

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
        #ifdef _OPENACC
        #pragma acc loop vector
        #endif
        #ifdef _OPENMP
        #pragma omp simd
        #endif
        for (int i = st; i < ed; i++) {
            x[i] = rhs[i] - a[i] * xl - c[i] * key;
        }
    }

    int m = n / s;
    return m * ((14 * s - 10) * fllog2(s) + 14) // stage 1
            + 14 * m * fllog2(m) // stage 2
            + m * (4 * s + 1); // stage 3
}


/**
 * @brief TPR STAGE 2
 *
 */
void TPR::tpr_stage2() {
    #pragma acc kernels present(this)
    {
        // Update by E_{st} and E_{ed} copy E_{ed} for stage 2 use
        #pragma acc loop independent
        for (int st = 0; st < this->n; st += s) {
            // EquationInfo eqi = update_uppper_no_check(st, ed);
            int k = st, kr = st + s - 1;
            int eqi_dst = 2 * st / s;
            real ak = a[k];
            real akr = a[kr];
            real ck = c[k];
            real ckr = c[kr];
            real rhsk = rhs[k];
            real rhskr = rhs[kr];

            real inv_diag_k = 1.0 / (1.0 - akr * ck);

            this->inter_a[eqi_dst] = inv_diag_k * ak;
            this->inter_c[eqi_dst] = -inv_diag_k * ckr * ck;
            this->inter_rhs[eqi_dst] = inv_diag_k * (rhsk - rhskr * ck);

            // Copy E_{ed}
            this->inter_a[eqi_dst + 1] = akr; // a.k.a. a[ed]
            this->inter_c[eqi_dst + 1] = ckr;
            this->inter_rhs[eqi_dst + 1] = rhskr;
        }

        // INTERMIDIATE STAGE
        {
            int len_inter = 2 * n / s;

            #pragma acc loop independent
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int i = 1; i < len_inter - 1; i += 2) {
                int k = i;
                int kr = i + 1;
                real ak = this->inter_a[k];
                real akr = this->inter_a[kr];
                real ck = this->inter_c[k];
                real ckr = this->inter_c[kr];
                real rhsk = this->inter_rhs[k];
                real rhskr = this->inter_rhs[kr];

                real inv_diag_k = 1.0 / (1.0 - akr * ck);

                int dst = i / 2;
                this->st2_a[dst] = inv_diag_k * ak;
                this->st2_c[dst] = -inv_diag_k * ckr * ck;
                this->st2_rhs[dst] = inv_diag_k * (rhsk - rhskr * ck);
            }

            this->st2_a[n / s - 1] = this->inter_a[len_inter - 1];
            this->st2_c[n / s - 1] = this->inter_c[len_inter - 1];
            this->st2_rhs[n / s - 1] = this->inter_rhs[len_inter - 1];
        }
    }


    PCR p = PCR(this->st2_a, nullptr, this->st2_c, this->st2_rhs, n / s);
    p.solve();
    // assert this->st2_rhs has the answer
    // copy back to TPR::x
    #pragma acc kernels present(this)
    {
        #pragma acc loop independent
        for (int i = s - 1; i < n - s; i += s) {
           this->x[i] = this->st2_rhs[i / s];
        }
        this->x[n - 1] = this->st2_rhs[n / s - 1];
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
    #pragma acc update host(this->x[:n])

    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};


#undef UNREACHABLE
