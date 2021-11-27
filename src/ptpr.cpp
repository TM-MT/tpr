#include <assert.h>

#include <array>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "lib.hpp"
#include "pcr.hpp"
#include "ptpr.hpp"
#include "tpr_perf.hpp"

#ifdef __GNUC__
#define UNREACHABLE __builtin_unreachable()
#else
#define UNREACHABLE
#endif

using namespace PTPR_Helpers;
using namespace tprperf;

/**
 * @brief      extend `x` to the length n' s.t. n' >= n, n' % s == 0
 *
 * @param      x      x[0:n] input array
 * @param[in]  n      the length of x
 * @param[in]  s      TPR parameter `s`
 * @param      new_x  The new x, x[0:n']
 *
 * @return     n'
 */
int PTPR_Helpers::add_padding(real *x, int n, int s, real **new_x) {
    if (n % s == 0) {
        *new_x = x;
        return n;
    } else {
        int new_n = (n / s + 1) * s;
        real *tmp;
        RMALLOC(tmp, new_n);

        for (int i = 0; i < n; i++) {
            tmp[i] = x[i];
        }
        for (int i = n; i < new_n; i++) {
            tmp[i] = 0.0f;
        }

        SAFE_DELETE(x);

        *new_x = tmp;
        return new_n;
    }
}

/**
 * @brief      set Tridiagonal System for PTPR
 *
 *             USAGE
 * @code{.cpp}
 *             PTPR t(n, s);
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
void PTPR::set_tridiagonal_system(real *a, real *c, real *rhs) {
    this->a = a;
    this->c = c;
    this->rhs = rhs;
}

/**
 * @brief      Initializer for PTPR()
 * @details    Call this function first to set up for PTPR
 *
 * @note       This function should call once.
 *
 * @param      n     size of given system. `n` should be power of 2
 * @param      s     size of a slice. `s` should be power of 2
 */
void PTPR::init(int n, int s) {
    auto format = std::string("PTPR_n_");
    tprperf::init(format.replace(5, 1, std::to_string(s)));

    assert(n % s == 0);
    this->n = n;
    this->s = s;
    this->m = n / s;

    // solver for stage 2
    this->st2solver.init(this->m);

    // allocation for answer
    RMALLOC(this->x, n + 1);
    for (int i = 0; i < n + 1; i++) {
        this->x[i] = 0;
    }
    this->x = &this->x[1];

    // allocation for stage 1 use
    RMALLOC(this->aa, n);
    RMALLOC(this->cc, n);
    RMALLOC(this->rr, n);
    // allocation for stage 2 use
    RMALLOC(this->st2_a, this->m);
    RMALLOC(this->st2_c, this->m);
    RMALLOC(this->st2_rhs, this->m);
    // allocation for bkup use
    RMALLOC(this->bkup_a, this->m);
    RMALLOC(this->bkup_c, this->m);
    RMALLOC(this->bkup_rhs, this->m);

    // NULL CHECK
    assert(this->x != nullptr);
    assert(this->aa != nullptr);
    assert(this->cc != nullptr);
    assert(this->aa != nullptr);
    assert(this->st2_a != nullptr);
    assert(this->st2_c != nullptr);
    assert(this->st2_rhs != nullptr);
    assert(this->bkup_a != nullptr);
    assert(this->bkup_c != nullptr);
    assert(this->bkup_rhs != nullptr);
}

/**
 * @brief      solve
 *
 * @return     num of float operation
 */
int PTPR::solve() {
    int fp_st1 = m * (14 * s * fllog2(s));
    int fp_st2 = 14 * m + (m - 1) * 14 + (14 * m * fllog2(m));
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

    // STAGE 3
    tprperf::start(tprperf::Labels::st3);
    tpr_stage3();
    tprperf::stop(tprperf::Labels::st3, static_cast<double>(fp_st3));

    return fp_st1 + fp_st2 + fp_st3;
}

/**
 * @brief      PTPR STAGE 1
 */
void PTPR::tpr_stage1() {
#pragma omp parallel for schedule(static)
    // for each slice
    for (int st = 0; st < this->n; st += s) {
        for (int p = 1; p <= static_cast<int>(log2(s)); p += 1) {
            int u = pow2(p - 1);
            // tpr_stage1(st, st + s - 1);
            int ed = st + s - 1;

#pragma omp simd
            for (int i = st; i < st + u; i++) {
                assert(i + u <= ed);

                // from update_uppper_no_check(i, i + u);
                int k = i;
                int kr = i + u;

                real inv_diag_k = 1.0f / (1.0f - a[kr] * c[k]);

                aa[k] = inv_diag_k * a[k];
                cc[k] = -inv_diag_k * c[kr] * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kr] * c[k]);
            }

#pragma omp simd
            for (int i = st + u; i <= ed - u; i++) {
                assert(st <= i - u);
                assert(i + u <= ed);

                // from update_no_check(i - u , i, i + u);
                int kl = i - u;
                int k = i;
                int kr = i + u;
                real inv_diag_k = 1.0f / (1.0f - c[kl] * a[k] - a[kr] * c[k]);

                aa[k] = -inv_diag_k * a[kl] * a[k];
                cc[k] = -inv_diag_k * c[kr] * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
            }

#pragma omp simd
            for (int i = ed - u + 1; i <= ed; i++) {
                assert(st <= i - u);

                // form update_lower_no_check(i - u, i);
                int kl = i - u;
                int k = i;
                real inv_diag_k = 1.0f / (1.0f - c[kl] * a[k]);

                aa[k] = -inv_diag_k * a[kl] * a[k];
                cc[k] = inv_diag_k * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k]);
            }

            // patch
            for (int i = st; i <= st + s - 1; i++) {
                this->a[i] = aa[i];
                this->c[i] = cc[i];
                this->rhs[i] = rr[i];
            }
        }  // end p

        // Update by E_{st} and E_{ed}, make bkup for stage 3 use.
        {
            // EquationInfo eqi = update_uppper_no_check(st, ed);
            int k = st, kr = st + s - 1;
            real ak = a[k];
            real akr = a[kr];
            real ck = c[k];
            real ckr = c[kr];
            real rhsk = rhs[k];
            real rhskr = rhs[kr];

            real inv_diag_k = 1.0f / (1.0f - akr * ck);

            // make bkup for stage 3 use
            int dst = st / this->s;
            this->bkup_a[dst] = this->a[k];
            this->bkup_c[dst] = this->c[k];
            this->bkup_rhs[dst] = this->rhs[k];

            this->a[k] = inv_diag_k * ak;
            this->c[k] = -inv_diag_k * ckr * ck;
            this->rhs[k] = inv_diag_k * (rhsk - rhskr * ck);
        }
    }
}

/**
 * @brief      PTPR Intermediate Stage
 */
void PTPR::tpr_inter() {
#pragma omp simd
    for (int i = this->s - 1; i < this->n - 1; i += this->s) {
        int k = i;
        int kr = i + 1;
        real ak = this->a[k];
        real akr = this->a[kr];
        real ck = this->c[k];
        real ckr = this->c[kr];
        real rhsk = this->rhs[k];
        real rhskr = this->rhs[kr];

        real inv_diag_k = 1.0f / (1.0f - akr * ck);

        int dst = i / this->s;
        this->st2_a[dst] = inv_diag_k * ak;
        this->st2_c[dst] = -inv_diag_k * ckr * ck;
        this->st2_rhs[dst] = inv_diag_k * (rhsk - rhskr * ck);
    }

    this->st2_a[this->m - 1] = this->a[this->n - 1];
    this->st2_c[this->m - 1] = this->c[this->n - 1];
    this->st2_rhs[this->m - 1] = this->rhs[this->n - 1];
}

/**
 * @brief      PTPR STAGE 2
 */
void PTPR::tpr_stage2() {
    // assert tpr_inter() have been called and st2_* have valid input.
    this->st2solver.set_tridiagonal_system(this->st2_a, nullptr, this->st2_c,
                                           this->st2_rhs);
    this->st2solver.solve();

    // this->st2solver.get_ans(this->st2_rhs);
    // assert this->st2_rhs has the answer
    // copy back to PTPR::x
    for (int i = s - 1; i < n; i += s) {
        this->x[i] = this->st2_rhs[i / s];
    }
}

/**
 * @brief      PTPR STAGE 3
 */
void PTPR::tpr_stage3() {
#pragma omp parallel for schedule(static)
    for (int st = 0; st < this->n; st += s) {
        // restore bkup
        int src = st / this->s;
        this->a[st] = this->bkup_a[src];
        this->c[st] = this->bkup_c[src];
        this->rhs[st] = this->bkup_rhs[src];

        // from tpr_stage3(st, st + s - 1);
        int ed = st + s - 1;
        // x[-1] should be 0.0

        real key = 1.0f / c[ed] * (rhs[ed] - a[ed] * x[st - 1] - x[ed]);
        if (c[ed] == 0.0f) {
            key = 0.0f;
        }

#pragma omp simd
        // x[ed] is known
        for (int i = st; i < ed; i++) {
            x[i] = rhs[i] - a[i] * x[st - 1] - c[i] * key;
        }
    }
}

/// Update E_k by E_{kl}, E_{kr}
EquationInfo PTPR::update_no_check(int kl, int k, int kr) {
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

    real inv_diag_k = 1.0f / (1.0f - ckl * ak - akr * ck);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = -inv_diag_k * akl * ak;
    eqi.c = -inv_diag_k * ckr * ck;
    eqi.rhs = inv_diag_k * (rhsk - rhskl * ak - rhskr * ck);

    return eqi;
}

/// Update E_k by E_{kr}
EquationInfo PTPR::update_uppper_no_check(int k, int kr) {
    assert(0 <= k && k < kr && kr < n);
    real ak = a[k];
    real akr = a[kr];
    real ck = c[k];
    real ckr = c[kr];
    real rhsk = rhs[k];
    real rhskr = rhs[kr];

    real inv_diag_k = 1.0f / (1.0f - akr * ck);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = inv_diag_k * ak;
    eqi.c = -inv_diag_k * ckr * ck;
    eqi.rhs = inv_diag_k * (rhsk - rhskr * ck);

    return eqi;
}

/// Update E_k by E_{kl}
EquationInfo PTPR::update_lower_no_check(int kl, int k) {
    assert(0 <= kl && kl < k && k < n);
    real ak = a[k];
    real akl = a[kl];
    real ck = c[k];
    real ckl = c[kl];
    real rhskl = rhs[kl];
    real rhsk = rhs[k];

    real inv_diag_k = 1.0f / (1.0f - ckl * ak);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = -inv_diag_k * akl * ak;
    eqi.c = inv_diag_k * ck;
    eqi.rhs = inv_diag_k * (rhsk - rhskl * ak);

    return eqi;
}

/**
 * @brief      get the answer
 *
 * @param      x     x[0:n] for the solution.
 *
 * @return     num of float operation
 */
int PTPR::get_ans(real *x) {
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};

#undef UNREACHABLE
