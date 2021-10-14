#include <gtest/gtest.h>

#include "PerfMonitor.h"
#include "cr.hpp"
#include "lib.hpp"
#include "main.hpp"
#include "pcr.hpp"
#include "pm.hpp"
#include "ptpr.hpp"
#include "system.hpp"
#include "tpr.hpp"

pm_lib::PerfMonitor pmcpp::pm = pm_lib::PerfMonitor();

class Examples : public ::testing::Test {
   public:
    const static int n = 1024;
    trisys::ExampleFixedInput *input = nullptr;

    real ans_array[n] = {
#include "ans1024.txt"
    };

   protected:
    void SetUp() override {
        input = new trisys::ExampleFixedInput(n);
        input->assign();
    }

    void TearDown() override { delete input; }

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
#pragma acc data copy( \
    input->sys.a[:n], input->sys.c[:n], input->sys.rhs[:n], input->sys.n)
    {
        CR cr(input->sys.a, input->sys.diag, input->sys.c, input->sys.rhs, n);
        cr.solve();
        cr.get_ans(input->sys.diag);
    }
    array_float_eq(ans_array, input->sys.diag);
    array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
}

TEST_F(Examples, PCRTest) {
#pragma acc data copy( \
    input->sys.a[:n], input->sys.c[:n], input->sys.rhs[:n], input->sys.n)
    {
        PCR pcr(input->sys.a, input->sys.diag, input->sys.c, input->sys.rhs,
                input->sys.n);
        pcr.solve();
        pcr.get_ans(input->sys.diag);
    }
    array_float_eq(ans_array, input->sys.diag);
    array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
}

TEST_F(Examples, TPRTest) {
    for (int s = 4; s <= n; s *= 2) {
        input->assign();
#pragma acc data copy(                                                      \
    input->sys.a[:n], input->sys.diag[:n], input->sys.c[:n], input->sys.rhs \
    [:n], input->sys.n)
        {
            TPR t(input->sys.a, input->sys.diag, input->sys.c, input->sys.rhs,
                  input->sys.n, s);
            t.solve();
            t.get_ans(input->sys.diag);
        }
        print_array(input->sys.diag, n);
        printf("\n");
    }
    array_float_eq(ans_array, input->sys.diag);
    array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
}

TEST_F(Examples, PTPRTest) {
    for (int s = 4; s <= n; s *= 2) {
        input->assign();
#pragma acc data copy(                                                      \
    input->sys.a[:n], input->sys.diag[:n], input->sys.c[:n], input->sys.rhs \
    [:n], input->sys.n)
        {
            PTPR t(input->sys.a, input->sys.diag, input->sys.c, input->sys.rhs,
                   input->sys.n, s);
            t.solve();
            t.get_ans(input->sys.diag);
        }
        print_array(input->sys.diag, n);
        printf("\n");
    }
    array_float_eq(ans_array, input->sys.diag);
    array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
}
