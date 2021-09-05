#include "lib.hpp"
#include "cr.hpp"


/**
 * @brief set Tridiagnoal System
 * @note [OpenACC] given arrays are exists on the device
 * 
 * @param a [description]
 * @param diag [description]
 * @param c [description]
 * @param rhs [description]
 */
void CR::set_tridiagonal_system(real *a, real *diag, real *c, real *rhs) {
    this->a = a;
    this->c = c;
    this->rhs = rhs;
    #pragma acc update device(this)
    #pragma acc enter data attach(this->a, this->c, this->rhs)
}


int CR::solve() {
    if (this->n > 1) {
        fr();
        bs();
        return 17 * this->n;
    } else {
        #pragma acc kernels present(this)
        {
            x[0] = rhs[0];
        }
        return 0;
    }
}

/**
 * @brief CR Forward Reduction
 * @return
 */
int CR::fr() {
    #pragma acc data present(a[:n], c[:n], rhs[:n], aa[:n], cc[:n], rr[:n], this)
    #pragma omp parallel
    for (int p = 0; p < fllog2(this->n) - 1; p++) {
        #pragma acc kernels
        {
            #pragma acc update device(p)
            int u = 1 << p;
            const int ux = 1 << (p+1);

            #pragma acc loop independent
            #pragma omp for simd schedule(static)
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
    #pragma acc data present(a[:n], c[:n], rhs[:n], aa[:n], cc[:n], rr[:n], this)
    {
        #pragma acc serial
        {
            int i = this->n / 2 - 1;
            int u = this->n / 2;
            real inv_det = 1.0 / (1.0 - c[i]*a[i+u]);

            x[i] = (rhs[i] - c[i]*rhs[i+u]) * inv_det;
            x[i+u] =  (rhs[i+u] - rhs[i]*a[i+u]) * inv_det;
        }

        #pragma omp parallel
        for (int k = fllog2(this->n) - 2; k >= 0; k--) {

            #pragma acc kernels
            {
                #pragma acc update device(k)
                int u = 1 << k;
                #pragma omp single
                {
                    int i = u - 1;
                    this->x[i] = rhs[i] - c[i] * x[i+u];
                }

                #pragma acc loop independent
                #pragma omp for simd
                for (int i = u - 1 + 2*u; i < this->n - u; i += 2*u) {
                    this->x[i] = rhs[i] - a[i] * x[i-u] - c[i] * x[i+u];
                }
            }
        }
    }

    return 0;
}

/**
 * @brief get the answer
 * 
 * @note [OpenACC] assert `*x` exists at the device
 * @param x x[0:n]
 */
int CR::get_ans(real *x) {
    #pragma acc parallel loop present(x[:n], this)
    for (int i = 0; i < this->n; i++) {
        x[i] = this->x[i];
    }
    return 0;
}
