#include <stdlib.h>

#include <iostream>

#include "main.hpp"
#include "ptpr.cuh"
#include "tpr.cuh"

int main() {
    int n = 8192;
    struct TRIDIAG_SYSTEM *sys =
        (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
    setup(sys, n);
    TPR_CU::TPR_ANS ans1(n), ans2(n);
    for (int s = 128; s <= std::min(n, 1024); s *= 2) {
        assign(sys);
        ans1.s = s;
        TPR_CU::tpr_cu(sys->a, sys->c, sys->rhs, ans1.x, n, s);

        if (s > 128 && ans1 != ans2) {
            std::cout << "TPR(" << ans1.n << "," << ans1.s << ") and TPR("
                      << ans2.n << "," << ans2.s << ") has different answer.\n";
            std::cout << "TPR(" << ans1.n << "," << ans1.s << ")\n";
            ans1.display(std::cout);
            std::cout << "TPR(" << ans2.n << "," << ans2.s << ")\n";
            ans2.display(std::cout);
        }
        ans2 = ans1;
    }

    assign(sys);
    if (n <= 1024) {
        // currently CR works in thread,
        TPR_CU::cr_cu(sys->a, sys->c, sys->rhs, ans1.x, n);
        ans1.display(std::cout);
    }

    clean(sys);
    free(sys);
}

int setup(struct TRIDIAG_SYSTEM *sys, int n) {
    sys->a = (real *)malloc(n * sizeof(real));
    sys->diag = (real *)malloc(n * sizeof(real));
    sys->c = (real *)malloc(n * sizeof(real));
    sys->rhs = (real *)malloc(n * sizeof(real));
    sys->n = n;

    return sys_null_check(sys);
}

int assign(struct TRIDIAG_SYSTEM *sys) {
    int n = sys->n;
    for (int i = 0; i < n; i++) {
        sys->a[i] = -1.0 / 6.0;
        sys->c[i] = -1.0 / 6.0;
        sys->diag[i] = 1.0;
        sys->rhs[i] = 1.0 * (i + 1);
    }
    sys->a[0] = 0.0;
    sys->c[n - 1] = 0.0;

    return 0;
}

int clean(struct TRIDIAG_SYSTEM *sys) {
    for (auto p : {sys->a, sys->diag, sys->c, sys->rhs}) {
        free(p);
    }

    sys->a = nullptr;
    sys->diag = nullptr;
    sys->c = nullptr;
    sys->rhs = nullptr;

    return 0;
}

bool sys_null_check(struct TRIDIAG_SYSTEM *sys) {
    for (auto p : {sys->a, sys->diag, sys->c, sys->rhs}) {
        if (p == nullptr) {
            return false;
        }
    }
    return true;
}
