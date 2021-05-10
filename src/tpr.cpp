#include <assert.h>

#include "backward.hpp"
#include "lib.hpp"
#include "tpr.hpp"


int TPR::solve() {
    tpr_forward();
    tpr_backward();

    return 0;
}


void TPR::tpr_forward() {
    int n = this->n;
    int s = this->s;

    int k, u, p;
    
    // STAGE 1
    for (k = 1; k <= fllog2(s); k += 1) {
        u = pow2(k-1);
        p = 0;
        while (p < n) {
            for (int i = p; i < p + s; i += pow2(k)) {
                update_section(i, u);
            }
            // p >= 0, k >= 1 -> i >= 0
            for (int i=p+pow2(k)-1; i < p+s; i += pow2(k)) {
                update_section(i, u);
            }
            p += s;
        }
    }

    // k = k + 1;
    u = pow2(k - 1);
    p = 0;
    while (p < n) {
        update_uppper_no_check(p, p + (u - 1));
        p = p + s;
    }

    // INTERMIDIATE STAGE
    
    // for (int i = s; i <= n - s; i += s) {
    for (int i = s-1; i < n - s; i += s) {
        update_uppper_no_check(i, i + 1);
    }

    
    // STAGE 2
    int j = 0;
    while ( j < (int)log2((double)n / s) - 1) {
        u = pow2(k-1);

        for (int i = pow2(k)-1; i < n; i += pow2(k)) {
            update_global(i, u);
        }
        j += 1;
        k += 1;
    }    
}


void TPR::tpr_stage1(int st, int ed) {}


void TPR::update_section(int i, int u) {    
    int lb = i / s * s;
    int ub = lb + s;
    
    update_bd_check(i, u, lb, ub);
}

void TPR::update_global(int i, int u) {
    update_bd_check(i, u, 0, n);
}


/// check i-u, i+u \in [lb, ub),
void TPR::update_bd_check(int i, int u, int lb, int ub) {
    bool lb_check = lb <= i - u;
    bool ub_check = i + u < ub;
    
    if (lb_check && ub_check) {
        update_no_check(i - u, i, i + u);    
    } else if (ub_check) {
        update_uppper_no_check(i, i + u);
    } else if (lb_check) {
        update_lower_no_check(i - u, i);
    } else {
        // not happen
        assert(false);
    }    
}

/// Update E_i by E_{kl}, E_{kr}
void TPR::update_no_check(int kl, int k, int kr) {
    assert(0 <= kl && kl < k && k < kr && kr < n);
    real ai = a[k];
    real diagi = diag[k];
    real ci = c[k];
    real rhsi = rhs[k];

    real s1 = ai / diag[kl];
    real s2 = ci / diag[kr];

    a[k] = - a[kl] * s1;
    diag[k] = diagi - c[kl] * s1 - a[kr] * s2;
    c[k] = - c[kr] * s2;
    rhs[k] = rhsi - rhs[kl] * s1 - rhs[kr] * s2;
}


/// Update E_i by E_{kr}
void TPR::update_uppper_no_check(int i, int kr) {
    assert(0 <= i && i < kr && kr < n);

    real s2 = c[i] / diag[kr];

    // no update for a[i]
    diag[i] = diag[i] - a[kr] * s2;
    c[i] = -c[kr] * s2;
    rhs[i] = rhs[i] - rhs[kr] * s2;
}

/// Update E_i by E_{kl}
void TPR::update_lower_no_check(int kl, int i) {
    assert(0 <= kl && kl < i && i < n);

    real s1 = a[i] / diag[kl];

    a[i] = - a[kl] * s1;
    diag[i] = diag[i] - c[kl] * s1;
    // no update for c[i]
    rhs[i] = rhs[i] - rhs[kl] * s1;
}


void TPR::tpr_backward() {
    int capital_i = n / 2;
    int m = n / s;

    int u = capital_i;
    int i = capital_i-1;

    // STAGE 2 (continue)
    x[i] = (diag[i+u]*rhs[i] - c[i]*rhs[i+u]) / (diag[i+u]*diag[i] - c[i]*a[i+u]);
    x[i+u] = (rhs[i+u]*diag[i] - rhs[i]*a[i+u]) / (diag[i+u]*diag[i] - c[i]*a[i+u]);
    

    int j = 0;
    while (j < fllog2(m) - 1) {
        capital_i /= 2;
        u /= 2;
    
        assert(u > 0);
        for (i = capital_i - 1; i < n; i += 2*u) {
            x[i] = (rhs[i] - a[i]*x[i-u] - c[i]*x[i+u]) / diag[i];
        }

        j += 1;
    }
    

    for (i = 0; i < n; i += 1) {
        if (i % 2 == 0) {
            replace(i, fllog2(s) + 1);
        } else {
            // replace(i, 0); 
        }
    }

    // STAGE 3
    // for (j = 0; j < fllog2(s); j += 1) {
    while (u >= 2) {
        capital_i = capital_i / 2;
        u = u / 2;
        int p = 0;

        assert(u > 0);
        assert(capital_i > 0);
        while (p < n) {
            for (i = p + capital_i - 1; i < p + s; i += 2*u) {
                x[i] = (rhs[i] - a[i] * x[i-u] - c[i]*x[i+u]) / diag[i];
            }
            p += s;
        }
    }
}


/// changes the E_i by x[j]
void TPR::replace(int i, int j) {
    assert(0 <= i && i < n);
    assert(0 <= j && j < n);
    assert(diag[i] != 0.0);

    x[i] = (rhs[i] - c[i] * x[j]) / diag[i];
}


int TPR::get_ans(real *x) {
    for (int i = 0; i < n; i++) {
        x[i] = this->x[i];
    }
    return 0;
};
