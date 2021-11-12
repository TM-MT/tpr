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

/**
 * @brief INDEX CONVERTER FOR EXTENDED ARRAYS SUCH AS `this->a, this->c,
 * this->rhs`
 *
 * @param  index
 * @return index
 */
#define I2EXI(i) ((i) / this->s * this->sl + this->s + ((i) % this->s))

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
    this->a = extend_input_array(a);
    this->c = extend_input_array(c);
    this->rhs = extend_input_array(rhs);

    assert(this->a != nullptr);
    assert(this->c != nullptr);
    assert(this->rhs != nullptr);

    // for stage 1 use
    for (int mm = 0; mm < this->m; mm++) {
        // a[kl] = -1 for k - u < I2EXI(st)
        for (int i = 0; i < this->s; i++) {
            assert(mm * this->sl + i <
                   this->m * this->sl);  // must be less than array length
            assert(this->a[mm * this->sl + i] ==
                   0.0);  // there should be no data
            this->a[mm * this->sl + i] = -1.0;
        }
        // c[kr] = -1 for k + u > I2EXI(st)
        for (int i = 2 * this->s; i < 3 * this->s; i++) {
            assert(mm * this->sl + i < this->m * this->sl);
            assert(this->c[mm * this->sl + i] == 0.0);
            this->c[mm * this->sl + i] = -1.0;
        }
    }
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
    this->sl = 3 * s;

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

    // NULL CHECK
    assert(this->x != nullptr);
    assert(this->st2_a != nullptr);
    assert(this->st2_c != nullptr);
    assert(this->st2_rhs != nullptr);
    assert(this->bkup_a != nullptr);
    assert(this->bkup_c != nullptr);
    assert(this->bkup_rhs != nullptr);
}

/**
 * @brief extend input array
 *
 * @note use `this->n, this->s, this->m, this->sl`
 *
 * @param p [description]
 * @return [description]
 */
real *TPR::extend_input_array(real *p) {
    // this->sl > this->s
    real *RMALLOC(ret, m * this->sl);

    // Initialize
    for (int i = 0; i < m * this->sl; i++) {
        ret[i] = 0.0;
    }

    // copy p -> ret
    for (int i = 0; i < n; i++) {
        ret[I2EXI(i)] = p[i];
    }

    return ret;
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

    tpr_inter();

    // STAGE 2
    tprperf::start(tprperf::Labels::st2);
    tpr_stage2();
    tprperf::stop(tprperf::Labels::st2, static_cast<double>(fp_st2));

    tprperf::start(tprperf::Labels::st3);
    st3_replace();
    // TPR Stage 3
    tpr_stage3();
    tprperf::stop(tprperf::Labels::st3, static_cast<double>(fp_st3));

    return fp_st1 + fp_st2 + fp_st3;
}

/**
 * @brief      TPR stage 1
 */
void TPR::tpr_stage1() {
#pragma omp parallel for schedule(static)
    // Make Backup for Stage 3 use
    for (int st = 0; st < this->n; st += this->s) {
        // mk_bkup_init(st, st + this->s - 1);
        for (int i = st; i <= st + this->s - 1; i += 2) {
            int src = I2EXI(i);
            bkup_a[i] = a[src];
            bkup_c[i] = c[src];
            bkup_rhs[i] = rhs[src];
        }

        // TPR Stage 1
        for (int p = 1; p <= static_cast<int>(log2(s)); p += 1) {
            int u = pow2(p - 1);
            int p2k = pow2(p);
            // tpr_stage1(st, st + s - 1);
            int ed = st + s - 1;

#pragma omp simd
            for (int i = st; i <= ed - u; i += p2k) {
                assert(0 <= I2EXI(i) - u);
                assert(I2EXI(i) + u <= this->n * 3);

                // from update_no_check(i - u , i, i + u);
                int k = I2EXI(i);
                int kl = k - u;
                int kr = k + u;
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                aa[i] = -inv_diag_k * a[kl] * a[k];
                cc[i] = -inv_diag_k * c[kr] * c[k];
                rr[i] = inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
            }

#pragma omp simd
            for (int i = st + p2k - 1; i <= ed; i += p2k) {
                // from update_no_check(i - u , i, i + u);
                int k = I2EXI(i);
                int kl = k - u;
                int kr = k + u;
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                aa[i] = -inv_diag_k * a[kl] * a[k];
                cc[i] = -inv_diag_k * c[kr] * c[k];
                rr[i] = inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
            }

            // patch
            for (int i = st; i <= ed; i += p2k) {
                int dst = I2EXI(i);
                this->a[dst] = aa[i];
                this->c[dst] = cc[i];
                this->rhs[dst] = rr[i];
            }
            for (int i = st + p2k - 1; i <= ed; i += p2k) {
                int dst = I2EXI(i);
                this->a[dst] = aa[i];
                this->c[dst] = cc[i];
                this->rhs[dst] = rr[i];
            }
        }

        // Make Backup for stage 3 use
        // mk_bkup_st1(st, st + this->s - 1);
        for (int i = st + 1; i <= st + this->s - 1; i += 2) {
            int src = I2EXI(i);
            bkup_a[i] = a[src];
            bkup_c[i] = c[src];
            bkup_rhs[i] = rhs[src];
        }

        // Update by E_{st} and E_{ed}
        {
            // EquationInfo eqi = update_uppper_no_check(st, ed);
            int k = I2EXI(st), kr = I2EXI(st + s - 1);
            real ak = a[k];
            real akr = a[kr];
            real ck = c[k];
            real ckr = c[kr];
            real rhsk = rhs[k];
            real rhskr = rhs[kr];

            real inv_diag_k = 1.0 / (1.0 - akr * ck);

            this->a[k] = inv_diag_k * ak;
            this->c[k] = -inv_diag_k * ckr * ck;
            this->rhs[k] = inv_diag_k * (rhsk - rhskr * ck);
        }
    }
}

/**
 * @brief      TPR Intermediate Stage
 */
void TPR::tpr_inter() {
#pragma omp simd
    for (int i = this->s - 1; i < this->n - 1; i += this->s) {
        int k = I2EXI(i);
        int kr = I2EXI(i + 1);
        real ak = this->a[k];
        real akr = this->a[kr];
        real ck = this->c[k];
        real ckr = this->c[kr];
        real rhsk = this->rhs[k];
        real rhskr = this->rhs[kr];

        real inv_diag_k = 1.0 / (1.0 - akr * ck);

        int dst = i / this->s;
        this->st2_a[dst] = inv_diag_k * ak;
        this->st2_c[dst] = -inv_diag_k * ckr * ck;
        this->st2_rhs[dst] = inv_diag_k * (rhsk - rhskr * ck);
    }

    this->st2_a[this->m - 1] = this->a[I2EXI(this->n - 1)];
    this->st2_c[this->m - 1] = this->c[I2EXI(this->n - 1)];
    this->st2_rhs[this->m - 1] = this->rhs[I2EXI(this->n - 1)];
}

/**
 * @brief      TPR STAGE 2
 */
void TPR::tpr_stage2() {
    // assert tpr_inter() have been called and st2_* have valid input.
    this->st2solver.set_tridiagonal_system(this->st2_a, nullptr, this->st2_c,
                                           this->st2_rhs);
    this->st2solver.solve();
    this->st2solver.get_ans(this->st2_rhs);

    // copy back
    // this->st2_rhs has the answer
    for (int i = 0; i < this->m; i++) {
        x[(i + 1) * this->s - 1] = this->st2_rhs[i];
    }
}

void TPR::tpr_stage3() {
#pragma omp parallel for
    for (int st = 0; st < this->n; st += s) {
        for (int p = fllog2(s) - 1; p >= 0; p--) {
            // tpr_stage3(st, st + s - 1);
            int ed = st + this->s - 1;
            int u = pow2(p);

            assert(u > 0);

            {
                int i = st + u - 1;
                int src = I2EXI(i);
                real x_u;
                if (i - u < 0) {
                    x_u = 0.0;
                } else {
                    x_u = x[i - u];
                }
                this->x[i] = rhs[src] - a[src] * x_u - c[src] * x[i + u];
            }

#pragma omp simd
            // update x[i]
            for (int i = st + u - 1 + 2 * u; i <= ed; i += 2 * u) {
                int src = I2EXI(i);
                assert(i - u >= 0);
                assert(i + u < this->n);
                assert(0 <= src && src < 3 * this->n);

                this->x[i] = rhs[src] - a[src] * x[i - u] - c[src] * x[i + u];
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
    for (int mm = 0; mm < this->m; mm++) {
        int src_base = mm * this->s;
        int dst_base = mm * this->sl;
        for (int i = 0; i < this->s; i++) {
            int src = src_base + i;
            int dst = dst_base + i;
            this->a[dst] = this->bkup_a[src];
            this->c[dst] = this->bkup_c[src];
            this->rhs[dst] = this->bkup_rhs[src];
        }
    }
}

/**
 * @brief      get the answer
 *
 * @param      x     x[0:n] the solution vector
 *
 * @return     num of float operation
 */
int TPR::get_ans(real *x) {
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};

#undef UNREACHABLE
