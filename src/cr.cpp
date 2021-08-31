#include "lib.hpp"
#include "cr.hpp"


int CR::solve() {
    if (this->n > 1) {
        fr();
        bs();
        return 17 * this->n;
    } else {
        x[0] = rhs[0];
        return 0;
    }
}

/**
 * @brief CR Forward Reduction
 * @return
 */
int CR::fr() {
    for (int p = 0; p < fllog2(this->n) - 1; p++) {
        #pragma acc kernels present(a[:n], c[:n], rhs[:n], aa[:n], cc[:n], rr[:n], this)
        #pragma omp parallel
        {
            #pragma acc update device(p)
            int u = 1 << p;
            int ux = 1 << (p+1);

            #pragma acc loop independent
            #pragma omp for schedule(static)
            for (int i = ux - 1; i < this->n - ux; i += ux) {
                // update(i, u)
                int kl = i - u;
                int k = i;
                int kr = i + u;
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k] - a[kr] * c[k]);

                aa[k] = - inv_diag_k * a[kl] * a[k];
                cc[k] = - inv_diag_k * c[kr] * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k] - rhs[kr] * c[k]);
            }

            // update_upper_no_check(i, i - u)
            #pragma omp single
            {
                int k = this->n - 1;
                int kl = k - u;
                real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k]);

                aa[k] = - inv_diag_k * a[kl] * a[k];
                cc[k] = inv_diag_k * c[k];
                rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k]);
            }

            #pragma acc loop independent
            #pragma omp for schedule(static)
            for (int i = ux - 1; i < this->n; i += ux) {
                a[i] = aa[i];
                c[i] = cc[i];
                rhs[i] = rr[i];
            }
        }
    }

    return 0;
}


/**
 * @brief CR Backward Subtitution
 * @return
 */
int CR::bs() {
    #pragma acc serial default(present)
    {
        int i = this->n / 2 - 1;
        int u = this->n / 2;
        real inv_det = 1.0 / (1.0 - c[i]*a[i+u]);

        x[i] = (rhs[i] - c[i]*rhs[i+u]) * inv_det;
        x[i+u] =  (rhs[i+u] - rhs[i]*a[i+u]) * inv_det;
    }

    for (int k = fllog2(this->n) - 2; k >= 0; k--) {
        int u = 1 << k;

        #pragma acc kernels default(present)
        {
            {
                int i = u - 1;
                this->x[i] = rhs[i] - c[i] * x[i+u];
            }

            #pragma acc loop independent
            #pragma omp parallel for
            for (int i = u - 1 + 2*u; i < this->n - u; i += 2*u) {
                this->x[i] = rhs[i] - a[i] * x[i-u] - c[i] * x[i+u];
            }
        }
    }

    return 0;
}


int CR::get_ans(real *x) {
    #ifdef _OPENACC
    #pragma acc update host(this->x[:n])
    #endif

    #ifdef _OPENMP
    #pragma omp simd
    #endif
    for (int i = 0; i < this->n; i++) {
        x[i] = this->x[i];
    }
    return 0;
}
