#include <assert.h>

#include <array>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "cr.hpp"
#include "lib.hpp"
#include "tpr.hpp"
#include "tpr_perf.hpp"

#ifdef __GNUC__
#define UNREACHABLE __builtin_unreachable()
#else
#define UNREACHABLE
#endif

using namespace TPR_Helpers;
using namespace tprperf;

/**
 * @brief      set Tridiagonal System for TPR
 *
 *             USAGE
 * @code{.cpp}
 *             TPR t(n, s);
 *             t.set_tridiagonal_system(a, c, rhs);
 *             t.solve();
 *             t.get_ans(x);
 * @endcode
 *
 * @param[in]  a     a[0:n] The subdiagonal elements of A. Assert a[0] == 0.0
 * @param[in]  c     c[0:n] The superdiagonal elements of A. Assert c[n-1] ==
 *                   0.0
 * @param[in]  rhs   rhs[0:n] The right-hand-side of the equation.
 */
void TPR::set_tridiagonal_system(real *a, real *c, real *rhs) {
    this->a = a;
    this->c = c;
    this->rhs = rhs;
#ifdef _OPENACC
#pragma acc enter data copyin(this->a[:n], this->c[:n], this->rhs[:n])
#pragma acc update device(this)
#endif
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
    auto format = std::string("TPR_n_");
    tprperf::init(format.replace(4, 1, std::to_string(s)));

    this->n = n;
    this->s = s;
    this->m = n / s;

    // solver for stage 2
    this->st2solver.init(this->m);

    // allocation for answer
    RMALLOC(this->x, n);
    // allocation for backup
    RMALLOC(this->bkup_a, n);
    RMALLOC(this->bkup_c, n);
    RMALLOC(this->bkup_rhs, n);
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
#ifdef _OPENACC
#pragma acc enter data copyin(this)
#pragma acc enter data create(aa[:n], cc[:n], rr[:n])
#pragma acc enter data create(x[:n])
#pragma acc enter data create(bkup_a[:n], bkup_c[:n], bkup_rhs[:n])
#pragma acc enter data create(st2_a[:n / s], st2_c[:n / s], st2_rhs[:n / s])
#pragma acc enter data create( \
    inter_a[:2 * n / s], inter_c[:2 * n / s], inter_rhs[:2 * n / s])
#endif

    // NULL CHECK
    {
        bool none_null = true;
        none_null = none_null && (this->x != nullptr);
        none_null = none_null && (this->st2_a != nullptr);
        none_null = none_null && (this->st2_c != nullptr);
        none_null = none_null && (this->st2_rhs != nullptr);
        none_null = none_null && (this->bkup_a != nullptr);
        none_null = none_null && (this->bkup_c != nullptr);
        none_null = none_null && (this->bkup_rhs != nullptr);
        none_null = none_null && (this->inter_a != nullptr);
        none_null = none_null && (this->inter_c != nullptr);
        none_null = none_null && (this->inter_rhs != nullptr);

        if (!none_null) {
            printf("[%s] FAILED TO ALLOCATE an array.\n", __func__);
            abort();
        }
    }
}

/**
 * @brief      solve
 *
 * @return     num of float operation
 */
int TPR::solve() {
    int fp_st1 = 28 * n - 14;
    int fp_st2 = 14 * m - 19 + 4 * m;
    int fp_st3 = m * 4 * (s - 1);

    // STAGE 1
    tprperf::start(tprperf::Labels::st1);
    tpr_stage1();
    tprperf::stop(tprperf::Labels::st1, static_cast<double>(fp_st1));

    // STAGE 2
    tprperf::start(tprperf::Labels::st2);
    tpr_stage2();
    tprperf::stop(tprperf::Labels::st2, static_cast<double>(fp_st2));

    st3_replace();
    tprperf::start(tprperf::Labels::st3);
    // TPR Stage 3
    tpr_stage3();
    tprperf::stop(tprperf::Labels::st3, static_cast<double>(fp_st3));

    return fp_st1 + fp_st2 + fp_st3;
}

/**
 * @brief      TPR stage 1
 */
void TPR::tpr_stage1() {
#pragma acc data present(this, aa[:n], cc[:n], rr[:n], a[:n], c[:n], rhs[:n])
    {
#pragma acc parallel loop collapse(2)
#pragma omp parallel for
        // Make Backup for Stage 3 use
        for (int st = 0; st < this->n; st += this->s) {
            // mk_bkup_init(st, st + this->s - 1);
            for (int i = st; i <= st + this->s - 1; i += 2) {
                bkup_a[i] = a[i];
                bkup_c[i] = c[i];
                bkup_rhs[i] = rhs[i];
            }
        }

        // TPR Stage 1
        for (int p = 1; p <= static_cast<int>(log2(s)); p += 1) {
            int u = pow2(p - 1);
            int p2k = pow2(p);
#pragma acc kernels
#pragma acc loop independent
#pragma omp parallel for schedule(static)
            for (int st = 0; st < this->n; st += s) {
                // tpr_stage1(st, st + s - 1);
                int ed = st + s - 1;

                // update_uppper_no_check(st, st + u);
                {
                    int k = st;
                    int kr = st + u;
                    real inv_diag_k = 1.0 / (1.0 - a[kr] * c[k]);

                    aa[k] = inv_diag_k * a[k];
                    cc[k] = -inv_diag_k * c[kr] * c[k];
                    rr[k] = inv_diag_k * (rhs[k] - rhs[kr] * c[k]);
                }

#pragma acc loop independent
#pragma omp simd
                for (int i = st + p2k; i <= ed - u; i += p2k) {
                    assert(i + u <= ed);

                    // from update_no_check(i - u , i, i + u);
                    int kl = i - u;
                    int k = i;
                    int kr = i + u;
                    real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                    aa[k] = -inv_diag_k * a[kl] * a[k];
                    cc[k] = -inv_diag_k * c[kr] * c[k];
                    rr[k] =
                        inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
                }

#pragma acc loop independent
#pragma omp simd
                for (int i = st + p2k - 1; i <= ed - u; i += p2k) {
                    assert(st <= i - u);
                    assert(i + u <= ed);

                    // from update_no_check(i - u , i, i + u);
                    int kl = i - u;
                    int k = i;
                    int kr = i + u;
                    real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                    aa[k] = -inv_diag_k * a[kl] * a[k];
                    cc[k] = -inv_diag_k * c[kr] * c[k];
                    rr[k] =
                        inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
                }

                // update_lower_no_check(ed - u, ed);
                {
                    int kl = ed - u;
                    int k = ed;
                    real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k]);

                    aa[k] = -inv_diag_k * a[kl] * a[k];
                    cc[k] = inv_diag_k * c[k];
                    rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k]);
                }

#pragma acc loop independent
                // patch
                for (int i = st; i <= ed; i += p2k) {
                    this->a[i] = aa[i];
                    this->c[i] = cc[i];
                    this->rhs[i] = rr[i];
                }
#pragma acc loop independent
                for (int i = st + p2k - 1; i <= ed; i += p2k) {
                    this->a[i] = aa[i];
                    this->c[i] = cc[i];
                    this->rhs[i] = rr[i];
                }
            }
        }

#pragma acc parallel loop collapse(2)
#pragma omp parallel for
        // Make Backup for stage 3 use
        for (int st = 0; st < this->n; st += this->s) {
            // mk_bkup_st1(st, st + this->s - 1);
            for (int i = st + 1; i <= st + this->s - 1; i += 2) {
                bkup_a[i] = a[i];
                bkup_c[i] = c[i];
                bkup_rhs[i] = rhs[i];
            }
        }
    }
}

/**
 * @brief      TPR STAGE 2
 */
void TPR::tpr_stage2() {
#pragma acc kernels present(this)
    {
#pragma acc loop independent
#pragma omp simd
        // Update by E_{st} and E_{ed} copy E_{ed} for stage 2 use
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
            this->inter_a[eqi_dst + 1] = akr;  // a.k.a. a[ed]
            this->inter_c[eqi_dst + 1] = ckr;
            this->inter_rhs[eqi_dst + 1] = rhskr;
        }

        // INTERMIDIATE STAGE
        {
            int len_inter = 2 * n / s;

#pragma acc loop independent
#pragma omp simd
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

    this->st2solver.set_tridiagonal_system(this->st2_a, nullptr, this->st2_c,
                                           this->st2_rhs);
    this->st2solver.solve();
    this->st2solver.get_ans(this->st2_rhs);

#pragma acc kernels loop independent present(this)
    // copy back
    // this->st2_rhs has the answer
    for (int i = 0; i < this->m; i++) {
        x[(i + 1) * this->s - 1] = this->st2_rhs[i];
    }
}

void TPR::tpr_stage3() {
#pragma acc data present(this, a[:n], c[:n], rhs[:n], x[:n])
    for (int p = fllog2(s) - 1; p >= 0; p--) {
#pragma acc kernels loop independent
#pragma omp parallel for
        for (int st = 0; st < this->n; st += s) {
            // tpr_stage3(st, st + s - 1);
            int ed = st + this->s - 1;
            int u = pow2(p);

            assert(u > 0);

            {
                int i = st + u - 1;
                real x_u;
                if (i - u < 0) {
                    x_u = 0.0;
                } else {
                    x_u = x[i - u];
                }
                this->x[i] = rhs[i] - a[i] * x_u - c[i] * x[i + u];
            }

#pragma acc loop independent
#pragma omp simd
            // update x[i]
            for (int i = st + u - 1 + 2 * u; i <= ed; i += 2 * u) {
                assert(i - u >= st);
                assert(i + u <= ed);

                this->x[i] = rhs[i] - a[i] * x[i - u] - c[i] * x[i + u];
            }
        }
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
    eqi.a = -inv_diag_k * akl * ak;
    eqi.c = -inv_diag_k * ckr * ck;
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
 * @brief   subroutine for STAGE 3 REPLACE
 *
 * @note    make sure `bkup_*` are allocated and call mk_bkup_* functions
 */
void TPR::st3_replace() {
#pragma acc parallel loop present(this)
    for (int i = 0; i < this->n; i++) {
        this->a[i] = this->bkup_a[i];
        this->c[i] = this->bkup_c[i];
        this->rhs[i] = this->bkup_rhs[i];
    }
}

/**
 * @brief      get the answer
 *
 * @note       [OpenACC] assert `*x` exists on device
 *
 * @param      x     x[0:n] the solution vector
 *
 * @return     num of float operation
 */
int TPR::get_ans(real *x) {
#pragma acc parallel loop present(x[:n], this->x[:n])
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};

#undef UNREACHABLE
