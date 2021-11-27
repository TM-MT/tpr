#include <gtest/gtest.h>

#include <vector>

#include "PerfMonitor.h"
#include "lib.hpp"
#include "main.hpp"
#include "pcr.hpp"
#include "pm.hpp"
#include "ptpr.hpp"
#include "system.hpp"
#include "tpr.hpp"

pm_lib::PerfMonitor pmcpp::pm = pm_lib::PerfMonitor();

class RandomInput : public ::testing::Test {
   public:
    int ns[7] = {256, 257, 258, 259, 260, 768, 1024};

   protected:
    void SetUp() override {}

    void TearDown() override {}
};

/**
 * This test only check if PTPR works fine.
 * It does NOT check the answer.
 */
TEST_F(RandomInput, PTPR) {
    for (auto &n : ns) {
        trisys::ExampleRandom input(n);

        for (int s = 4; s <= n; s *= 2) {
            input.assign();
            PTPR_Helpers::add_padding(input.sys.a, input.sys.n, s,
                                      &input.sys.a);
            PTPR_Helpers::add_padding(input.sys.diag, input.sys.n, s,
                                      &input.sys.diag);
            PTPR_Helpers::add_padding(input.sys.c, input.sys.n, s,
                                      &input.sys.c);
            int new_n = PTPR_Helpers::add_padding(input.sys.rhs, input.sys.n, s,
                                                  &input.sys.rhs);

            PTPR t(input.sys.a, input.sys.diag, input.sys.c, input.sys.rhs,
                   new_n, s);
            t.solve();
            t.get_ans(input.sys.diag);
            print_array(input.sys.diag, n);
            printf("\n");
        }
    }
}
