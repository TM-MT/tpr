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
 * @brief set Tridiagonal System for PTPR
 *
 * USAGE
 * ```
 * PTPR t(n, s);
 * t.set_tridiagonal_system(a, c, rhs);
 * t.solve();
 * t.get_ans(x);
 * ```
 *
 * @param a
 * @param c
 * @param rhs
 */
void PTPR::set_tridiagonal_system(real *a, real *c, real *rhs) {
    this->a = a;
    this->c = c;
    this->rhs = rhs;
    #ifdef _OPENACC
    #pragma acc enter data copyin(a[:n], c[:n], rhs[:n])
    #pragma acc update device(this)
    #endif
}

/**
 * @brief Initializer for PTPR()
 * @details Call this function first to set up for PTPR
 *
 * @note This function should call once.
 *
 * @param n size of given system
 * @param s size of a slice. `s` should be power of 2
 */
void PTPR::init(int n, int s) {
    auto format = std::string("PTPR_n_");
    tprperf::init(format.replace(5, 1, std::to_string(s)));

    this->n = n;
    this->s = s;
    this->m = n / s;

    // solver for stage 2
    this->st2solver.init(this->m);

    // allocation for answer
    RMALLOC(this->x, n+1);
    for (int i = 0; i < n + 1; i++) {
        this->x[i] = 0;
    }
    this->x = &this->x[1];

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
    #pragma acc enter data copyin(this, this->n, this->s)
    #pragma acc enter data create(aa[:n], cc[:n], rr[:n])
    #pragma acc enter data copyin(x[-1:n+1])
    #pragma acc enter data create(st2_a[:n/s], st2_c[:n/s], st2_rhs[:n/s])
    #pragma acc enter data create(inter_a[:2*n/s], inter_c[:2*n/s], inter_rhs[:2*n/s])
    #endif

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
 * @brief      PTPR STAGE 1
 *
 * @param[in]  st     start index of equation that this function calculate
 * @param[in]  ed     end index of equation that this function calculate
 */
void PTPR::tpr_stage1(int st, int ed) {
}

/**
 * @brief solve
 * @return num of float operation
 */
int PTPR::solve() {
    int fp_st1 = m * (14 * s * fllog2(s));
    int fp_st2 = 14 * m + (m-1) * 14 + (14 * m * fllog2(m));
    int fp_st3 = m * 4 * (s - 1);

    // STAGE 1
    tprperf::start(tprperf::Labels::st1);
    #pragma acc data present(this, a[:n], c[:n], rhs[:n], aa[:n], cc[:n], rr[:n])
    for (int p = 1; p <= static_cast<int>(log2(s)); p += 1) {
        int u = pow2(p-1);

        #ifdef _OPENACC
        #pragma acc parallel num_gangs(this->m) vector_length(this->s) copyin(u)
        #pragma acc loop gang
        #endif
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int st = 0; st < this->n; st += s) {
            // tpr_stage1(st, st + s - 1);
            int ed = st + s - 1;

            #ifdef _OPENACC
            #pragma acc loop vector
            #endif
            #ifdef _OPENMP
            #pragma omp simd
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
            #pragma acc loop vector
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
            #pragma acc loop vector
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
        }
        #pragma acc parallel loop collapse(2) num_gangs(this->m) vector_length(this->s)
        for (int st = 0; st < this->n; st += s) {
            // patch
            for (int i = st; i <= st + s - 1; i++) {
                this->a[i] = aa[i];
                this->c[i] = cc[i];
                this->rhs[i] = rr[i];
            }
        }

    }
    tprperf::stop(tprperf::Labels::st1, static_cast<double>(fp_st1));

    // STAGE 2
    tprperf::start(tprperf::Labels::st2);
    tpr_stage2();
    tprperf::stop(tprperf::Labels::st2, static_cast<double>(fp_st2));

    // STAGE 3
    tprperf::start(tprperf::Labels::st3);
    #ifdef _OPENACC
    #pragma acc parallel num_gangs(this->m) vector_length(this->s-1) present(this, a[:n], c[:n], rhs[:n], x[:n])
    #pragma acc loop gang independent
    #endif
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int st = 0; st < this->n; st += s) {
        // from tpr_stage3(st, st + s - 1);
        int ed = st + s - 1;
        // x[-1] should be 0.0

        real key = 1.0 / c[ed] * (rhs[ed] - a[ed] * x[st-1] - x[ed]);
        if (c[ed] == 0.0) {
            key = 0.0;
        }

        // x[ed] is known
        #ifdef _OPENACC
        #pragma acc loop vector
        #endif
        #ifdef _OPENMP
        #pragma omp simd
        #endif
        for (int i = st; i < ed; i++) {
            x[i] = rhs[i] - a[i] * x[st-1] - c[i] * key;
        }
    }
    tprperf::stop(tprperf::Labels::st3, static_cast<double>(fp_st3));

    return fp_st1 + fp_st2 + fp_st3;
}


/**
 * @brief PTPR STAGE 2
 *
 */
void PTPR::tpr_stage2() {
    #ifdef _OPENACC
    #pragma acc kernels present(this)
    #endif
    {
        // Update by E_{st} and E_{ed} copy E_{ed} for stage 2 use
        #ifdef _OPENACC
        #pragma acc loop independent
        #endif
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

            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
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


    this->st2solver.set_tridiagonal_system(this->st2_a, nullptr, this->st2_c, this->st2_rhs);
    this->st2solver.solve();
    // this->st2solver.get_ans(this->st2_rhs);
    // assert this->st2_rhs has the answer
    // copy back to PTPR::x
    #pragma acc kernels loop independent present(this)
    for (int i = s - 1; i < n; i += s) {
       this->x[i] = this->st2_rhs[i / s];
    }
}

/**
 * @brief      PTPR STAGE 3
 *
 * @param[in]  st     start index of equation that this function calculate
 * @param[in]  ed     end index of equation that this function calculate
 */
void PTPR::tpr_stage3(int st, int ed) {
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

    real inv_diag_k = 1.0 / (1.0 - ckl * ak - akr * ck);

    EquationInfo eqi;
    eqi.idx = k;
    eqi.a = - inv_diag_k * akl * ak;
    eqi.c = - inv_diag_k * ckr * ck;
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

    real inv_diag_k = 1.0 / (1.0 - akr * ck);

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
int PTPR::get_ans(real *x) {
    #pragma acc kernels loop present(this, this->x[:n], x[:n])
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};


#undef UNREACHABLE
