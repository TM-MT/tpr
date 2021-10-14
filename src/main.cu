/**
 * @brief      Example main function for cuda program call
 *
 * @author     TM-MT
 * @date       2021
 */
#include <stdlib.h>

#include <iostream>

#include "main.hpp"
#include "ptpr.cuh"
#include "reference_cusparse.cuh"
#include "system.hpp"
#include "tpr.cuh"

int main() {
    int n = 8192;
    trisys::ExampleFixedInput input(n);

    TPR_ANS ans1(n), ans2(n);
    for (int s = 128; s <= std::min(n, 1024); s *= 2) {
        input.assign();
        ans1.s = s;
        TPR_CU::tpr_cu(input.sys.a, input.sys.c, input.sys.rhs, ans1.x, n, s);

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

    for (int s = 128; s <= std::min(n, 1024); s *= 2) {
        input.assign();
        ans1.s = s;
        PTPR_CU::ptpr_cu(input.sys.a, input.sys.c, input.sys.rhs, ans1.x, n, s);

        if (s > 128 && ans1 != ans2) {
            std::cout << "PTPR(" << ans1.n << "," << ans1.s << ") and PTPR("
                      << ans2.n << "," << ans2.s << ") has different answer.\n";
            std::cout << "PTPR(" << ans1.n << "," << ans1.s << ")\n";
            ans1.display(std::cout);
            std::cout << "PTPR(" << ans2.n << "," << ans2.s << ")\n";
            ans2.display(std::cout);
        }
        ans2 = ans1;
    }

    if (n <= 1024) {
        // currently CR works in thread,
        input.assign();
        TPR_CU::cr_cu(input.sys.a, input.sys.c, input.sys.rhs, ans1.x, n);
        ans1.display(std::cout);

        input.assign();
        PTPR_CU::pcr_cu(input.sys.a, input.sys.c, input.sys.rhs, ans1.x, n);
        ans1.display(std::cout);
    }

    input.assign();
    REFERENCE_CUSPARSE rfs(n);
    rfs.solve(input.sys.a, input.sys.c, input.sys.rhs, ans1.x, n);
    ans1.display(std::cout);
}
