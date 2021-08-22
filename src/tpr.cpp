#include <assert.h>
#include <array>

#ifdef __DEBUG__
#include "backward.hpp"
#endif

#include "lib.hpp"
#include "pcr.hpp"
#include "tpr.hpp"
#include "pm.hpp"
#include "PerfMonitor.h"


#ifdef __GNUC__
#define UNREACHABLE __builtin_unreachable()
#else
#define UNREACHABLE
#endif

/**
 * @brief INDEX CONVERTER FOR EXTENDED ARRAYS SUCH AS `this->a, this->c, this->rhs`
 * 
 * @param  index
 * @return index
 */
#define I2EXI(i)  ((i) / this->s * this->sl + this->s + ((i) % this->s))


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
    this->a = extend_input_array(a);
    this->c = extend_input_array(c);
    this->rhs = extend_input_array(rhs);

    if ((this->a == nullptr) 
        | (this->c == nullptr) 
        | (this->rhs == nullptr)) {
        printf("[%s] FAILED TO ALLOCATE an array.\n",
            __func__
            );
        abort();
    }

    // for stage 1 use
    for (int mm = 0; mm < this->m; mm++) {
        // a[kl] = -1 for k - u < I2EXI(st)
        for (int i = 0; i < this->s; i++) {
            assert(mm * this->sl + i < this->m * this->sl); // must be less than array length
            assert(this->a[mm * this->sl + i] == 0.0); // there should be no data
            this->a[mm * this->sl + i] = -1.0;
        }
        // c[kr] = -1 for k + u > I2EXI(st)
        for (int i = 2 * this->s; i < 3*this->s; i++) {
            assert(mm * this->sl + i < this->m * this->sl);
            assert(this->c[mm * this->sl + i] == 0.0);
            this->c[mm * this->sl + i] = -1.0;
        }
    }

    #ifdef _OPENACC
    #pragma acc enter data copyin(a[:m*sl], c[:m*sl], rhs[:m*sl])
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
void TPR::init(int n, int s, pm_lib::PerfMonitor *pm) {
    this->n = n;
    this->s = s;
    this->m = n / s;
    // this->sl = 2*pow2(fllog2(s)) + s;
    this->sl = 3 * s;
    #ifdef _OPENACC
    #pragma acc enter data copyin(this, this->n, this->s, this->m, this->sl)
    #endif

    this->pm = pm;

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
    #ifdef _OPENACC
    #pragma acc enter data create(aa[:n], cc[:n], rr[:n])
    #pragma acc enter data create(x[:n])
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

    // Initialize PerfMonitor and set labels
    for (unsigned long int i = 0; i < this->default_labels.size(); i++) {
        auto format = std::string("TPR_n_");
        auto gen_label = format.replace(4, 1, std::to_string(this->s))
                            .append(this->default_labels[i]);
        this->labels[i] = gen_label;
        this->pm->setProperties(gen_label, this->pm->CALC);
    }
}

/**
 * @brief extend input array
 * 
 * @note use `this->n, this->s, this->m, this->sl`
 * 
 * @param p [description]
 * @return [description]
 */
real* TPR::extend_input_array(real *p) {
    // this->sl > this->s
    real* RMALLOC(ret, m * this->sl);

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
    int fp_st1 = m * (14 * s * fllog2(s));
    int fp_st2 = 14 * m + (m-1) * 14 + (14 * m * fllog2(m));
    int fp_st3 = m * 4 * (s - 1);

    // STAGE 1
    this->pm->start(labels[0]);
    for (int k = 1; k <= static_cast<int>(log2(s)); k += 1) {
        const int u = pow2(k-1);
        const int s = this->s;

        #ifdef _OPENACC
        #pragma acc kernels present(this, a[:m*sl], c[:m*sl], rhs[:m*sl], aa[:n], cc[:n], rr[:n])
        #pragma acc loop independent
        #endif
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) shared(aa, cc, rr, a, c, rhs, u, s)
        #endif
        for (int st = 0; st < this->n; st += s) {
            // tpr_stage1(st, st + s - 1);
            int ed = st + s - 1;

            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            #ifdef _OPENMP
            #pragma omp simd
            #endif
            for (int i = st; i <= ed; i++) {
                // from update_no_check(i - u , i, i + u);
                int k = I2EXI(i);
                int kl = k - u;
                int kr = k + u;
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                aa[i] = - inv_diag_k * a[kl] * a[k];
                cc[i] = - inv_diag_k * c[kr] * c[k];
                rr[i] = inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
            }

            // patch
            #ifdef _OPENACC
            #pragma acc loop independent
            #endif
            for (int i = st; i <= ed; i++) {
                int dst = I2EXI(i);
                this->a[dst] = aa[i];
                this->c[dst] = cc[i];
                this->rhs[dst] = rr[i];
            }
        }
    }
    this->pm->stop(labels[0], static_cast<double>(fp_st1));

    // STAGE 2
    this->pm->start(labels[1]);
    tpr_stage2();
    this->pm->stop(labels[1], static_cast<double>(fp_st2));

    // STAGE 3
    this->pm->start(labels[2]);
    #ifdef _OPENACC
    #pragma acc parallel present(this, a[:m*sl], c[:m*sl], rhs[:m*sl], x[:n])
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
        {
            int ked = I2EXI(ed);
            if (c[ked] == 0.0) { // c[n] should be 0.0
                key = 0.0;
            } else {
                key = 1.0 / c[ked] * (rhs[ked] - a[ked] * xl - x[ed]);
            }
        }

        // x[ed] is known
        #ifdef _OPENACC
        #pragma acc loop vector
        #endif
        #ifdef _OPENMP
        #pragma omp simd
        #endif
        for (int i = st; i < ed; i++) {
            int k = I2EXI(i);
            x[i] = rhs[k] - a[k] * xl - c[k] * key;
        }
    }
    this->pm->stop(labels[2], static_cast<double>(fp_st3));

    return fp_st1 + fp_st2 + fp_st3;
}


/**
 * @brief TPR STAGE 2
 *
 */
void TPR::tpr_stage2() {
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
            int k = I2EXI(st), kr = I2EXI(st + s - 1);
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

    PCR p(this->st2_a, nullptr, this->st2_c, this->st2_rhs, n / s);
    p.solve();
    p.get_ans(this->st2_rhs);

    // copy back to TPR::x
    #ifdef _OPENACC
    #pragma acc kernels present(this, this->st2_rhs[:m])
    #endif
    {
        #ifdef _OPENACC
        #pragma acc loop independent
        #endif
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
    #ifdef _OPENACC
    #pragma acc update host(this->x[:n])
    #endif

    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};


#undef UNREACHABLE
