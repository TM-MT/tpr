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

/**
 * @brief      INDEX CONVERTER FOR EXTENDED ARRAYS SUCH AS `this->a, this->c,
 *             this->rhs`
 *
 * @param      i     Index
 *
 * @return     index
 */
#define I2EXI(i) ((i) / this->s * this->sl + this->s + ((i) % this->s))

using namespace PTPR_Helpers;
using namespace tprperf;

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

    this->n = n;
    this->s = s;
    this->m = n / s;
    this->sl = 3 * s;

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
 * @brief      extend input array
 *
 * @note       use `this->n, this->s, this->m, this->sl`
 *
 * @param      p     Input Array length of `this->n`
 *
 * @return     A pointer to the new array length of `this->m * this->sl`
 */
real *PTPR::extend_input_array(real *p) {
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
        const int src_base = I2EXI(st);
        for (int p = 1; p <= fllog2(s); p += 1) {
            int u = pow2(p - 1);
            // tpr_stage1(st, st + s - 1);
            int ed = st + s - 1;

#pragma omp simd
            for (int i = st; i <= ed; i++) {
                // from update_no_check(i - u , i, i + u);
                int k = src_base + i - st;
                assert(k == I2EXI(i));
                int kl = k - u;
                int kr = k + u;
                assert(0 <= kl);
                assert(kr < 3 * this->n);
                assert(I2EXI(st) - s < kl);
                assert(kr < I2EXI(ed) + s);
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                aa[i] = -inv_diag_k * a[kl] * a[k];
                cc[i] = -inv_diag_k * c[kr] * c[k];
                rr[i] = inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
            }

            // patch
            for (int i = st; i <= st + s - 1; i++) {
                int dst = src_base + i - st;
                this->a[dst] = aa[i];
                this->c[dst] = cc[i];
                this->rhs[dst] = rr[i];
            }
        }  // end p

        // Update by E_{st} and E_{ed}, make bkup for stage 3 use.
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
 * @brief      PTPR STAGE 2
 */
void PTPR::tpr_stage2() {
    // assert tpr_inter() have been called and st2_* have valid input.
    this->st2solver.set_tridiagonal_system(this->st2_a, nullptr, this->st2_c,
                                           this->st2_rhs);
    this->st2solver.solve();
    this->st2solver.get_ans(this->st2_rhs);

    // copy back to PTPR::x
    for (int i = 0; i < this->m; i++) {
        x[(i + 1) * this->s - 1] = this->st2_rhs[i];
    }
}

/**
 * @brief      PTPR STAGE 3
 */
void PTPR::tpr_stage3() {
#pragma omp parallel for schedule(static)
    for (int st = 0; st < this->n; st += s) {
        int exst = I2EXI(st);
        // restore bkup
        int src = st / this->s;
        this->a[exst] = this->bkup_a[src];
        this->c[exst] = this->bkup_c[src];
        this->rhs[exst] = this->bkup_rhs[src];

        // from tpr_stage3(st, st + s - 1);
        int ed = st + s - 1;
        int exed = exst + s - 1;
        // x[-1] should be 0.0

        real key = 1.0 / c[exed] * (rhs[exed] - a[exed] * x[st - 1] - x[ed]);
        if (c[exed] == 0.0) {
            key = 0.0;
        }

#pragma omp simd
        // x[ed] is known
        for (int i = st; i < ed; i++) {
            int exi = exst + i - st;
            assert(exi == I2EXI(i));
            x[i] = rhs[exi] - a[exi] * x[st - 1] - c[exi] * key;
        }
    }
}

/**
 * @brief      get the answer
 *
 * @param      x     x[0:n] for the solution vector
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
