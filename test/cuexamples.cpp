#include <gtest/gtest.h>

#include "lib.hpp"
#include "main.hpp"
#include "ptpr.cuh"
#include "system.hpp"
#include "tpr.cuh"

class CuExamples : public ::testing::Test {
   public:
    const static int n = 1024;
    trisys::ExampleFixedInput *input;

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

TEST_F(CuExamples, CRTest) {
    TPR_CU::cr_cu(input->sys.a, input->sys.c, input->sys.rhs, input->sys.diag,
                  n);
    array_float_eq(ans_array, input->sys.diag);
    array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
}

TEST_F(CuExamples, PCRTest) {
    PTPR_CU::pcr_cu(input->sys.a, input->sys.c, input->sys.rhs, input->sys.diag,
                    n);
    array_float_eq(ans_array, input->sys.diag);
    array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
}

TEST_F(CuExamples, TPRTest) {
    for (int s = 64; s <= 1024; s *= 2) {
        input->assign();
        TPR_CU::tpr_cu(input->sys.a, input->sys.c, input->sys.rhs,
                       input->sys.diag, n, s);
        print_array(input->sys.diag, n);
        array_float_eq(ans_array, input->sys.diag);
        array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
    }
}

TEST_F(CuExamples, PTPRTest) {
    for (int s = 64; s <= 1024; s *= 2) {
        input->assign();
        PTPR_CU::ptpr_cu(input->sys.a, input->sys.c, input->sys.rhs,
                         input->sys.diag, n, s);
        print_array(input->sys.diag, n);
        array_float_eq(ans_array, input->sys.diag);
        array_float_maxsqsum(ans_array, input->sys.diag, 1e-3);
    }
}
