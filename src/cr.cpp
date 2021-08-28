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


int CR::fr() {
    for (int p = 0; p < fllog2(this->n) - 1; p++) {
        int u = pow2(p);
        int i;

        for (i = pow2(p+1) - 1; i < this->n - pow2(p+1); i += pow2(p+1)) {
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
        {
            int k = this->n - 1;
            int kl = k - u;
            real inv_diag_k = 1.0 / (1.0 - c[kl] * a[k]);

            aa[k] = - inv_diag_k * a[kl] * a[k];
            cc[k] = inv_diag_k * c[k];
            rr[k] = inv_diag_k * (rhs[k] - rhs[kl] * a[k]);
        }

        for (int i = pow2(p+1) - 1; i < this->n; i += pow2(p+1)) {
            a[i] = aa[i];
            c[i] = cc[i];
            rhs[i] = rr[i];
        }
    }

    return 0;
}

int CR::bs() {
    {
        int i = this->n / 2 -1;
        int u = this->n / 2;
        real inv_det = 1.0 / (1.0 - c[i]*a[i+u]);

        x[i] = (rhs[i] - c[i]*rhs[i+u]) * inv_det;
        x[i+u] =  (rhs[i+u] - rhs[i]*a[i+u]) * inv_det;
    }

    for (int k = fllog2(this->n) - 2; k >= 0; k--) {
    	int u = pow2(k);

    	// use aa[:n] as new_x[:n]
    	{
	    	int i = pow2(k) - 1;
    		this->aa[i] = rhs[i] - c[i] * x[i+u];
    	}

    	for (int i = pow2(k) - 1 + pow2(k+1); i < this->n - u; i += pow2(k+1)) {
    		this->aa[i] = rhs[i] - a[i] * x[i-u] - c[i] * x[i+u];
    	}

    	for (int i = pow2(k) - 1; i < this->n - u; i += pow2(k+1)) {
			this->x[i] = this->aa[i];
    	}
    }

    return 0;
}


int CR::get_ans(real *x) {
	for (int i = 0; i < this->n; i++) {
		x[i] = this->x[i];
	}
	return 0;
}