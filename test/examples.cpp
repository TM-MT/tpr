#include <gtest/gtest.h>

#include "PerfMonitor.h"
#include "cr.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "pcr.hpp"
#include "pm.hpp"
#include "ptpr.hpp"
#include "tpr.hpp"

pm_lib::PerfMonitor pmcpp::pm = pm_lib::PerfMonitor();

class Examples : public ::testing::Test {
   public:
    struct TRIDIAG_SYSTEM *sys = nullptr;
    const static int n = 1024;

    real ans_array[n] = {
#include "ans1024.txt"
    };

   protected:
    void SetUp() override {
        sys = (struct TRIDIAG_SYSTEM *)malloc(sizeof(struct TRIDIAG_SYSTEM));
        setup(sys, n);

        assign(sys);
    }

    void TearDown() override {
        clean(sys);
        free(sys);
    }

    void array_float_eq(real *expect, real *actual) {
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(expect[i], actual[i], 1e-2)
                << "Expect " << expect[i] << " but got " << actual[i]
                << " at index " << i << "\n";
        }
    }

    void array_float_maxsqsum(real *expect, real *actual, real thre) {
        real maxsqsum = 0.0;
        int idx = -1;
        for (int i = 0; i < n; i++) {
            real d = powf(fabs(expect[i] - actual[i]), 2);
            if (d > maxsqsum) {
                maxsqsum = d;
                idx = i;
            }
        }

        EXPECT_NEAR(0.0, maxsqsum, thre) << "At index=" << idx << "\n";
    }
};

TEST_F(Examples, CRTest) {
#pragma acc data copy(sys->a[:n], sys->c[:n], sys->rhs[:n], sys->n)
    {
        CR cr(sys->a, sys->diag, sys->c, sys->rhs, sys->n);
        cr.solve();
        cr.get_ans(sys->diag);
    }
    array_float_eq(ans_array, sys->diag);
    array_float_maxsqsum(ans_array, sys->diag, 1e-3);
}

TEST_F(Examples, PCRTest) {
#pragma acc data copy(sys->a[:n], sys->c[:n], sys->rhs[:n], sys->n)
    {
        PCR pcr(sys->a, sys->diag, sys->c, sys->rhs, sys->n);
        pcr.solve();
        pcr.get_ans(sys->diag);
    }
    array_float_eq(ans_array, sys->diag);
    array_float_maxsqsum(ans_array, sys->diag, 1e-3);
}

TEST_F(Examples, TPRTest) {
    for (int s = 4; s <= n; s *= 2) {
        assign(sys);
#pragma acc data copy( \
    sys->a[:n], sys->diag[:n], sys->c[:n], sys->rhs[:n], sys->n)
        {
            TPR t(sys->a, sys->diag, sys->c, sys->rhs, sys->n, s);
            t.solve();
            t.get_ans(sys->diag);
        }
        print_array(sys->diag, n);
        printf("\n");
    }
    array_float_eq(ans_array, sys->diag);
    array_float_maxsqsum(ans_array, sys->diag, 1e-3);
}

TEST_F(Examples, PTPRTest) {
    for (int s = 4; s <= n; s *= 2) {
        assign(sys);
#pragma acc data copy( \
    sys->a[:n], sys->diag[:n], sys->c[:n], sys->rhs[:n], sys->n)
        {
            PTPR t(sys->a, sys->diag, sys->c, sys->rhs, sys->n, s);
            t.solve();
            t.get_ans(sys->diag);
        }
        print_array(sys->diag, n);
        printf("\n");
    }
    array_float_eq(ans_array, sys->diag);
    array_float_maxsqsum(ans_array, sys->diag, 1e-3);
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
