#pragma once
#include <vector>

#include "lib.hpp"

namespace trisys {
class TRIDIAG_SYSTEM {
   public:
    int n;
    real *a;
    real *diag;
    real *c;
    real *rhs;

    TRIDIAG_SYSTEM(int n);
    ~TRIDIAG_SYSTEM();
    bool null_check();
    TRIDIAG_SYSTEM(const TRIDIAG_SYSTEM &sys) { n = sys.n; };
};

/**
 * @brief      The base class for sample input
 */
class ExampleInput {
   public:
    TRIDIAG_SYSTEM sys;

    ExampleInput(int n) : sys(n){ assign(); };
    ~ExampleInput(){};
    virtual int assign() { return 0; };
    ExampleInput(const ExampleInput &ei) : sys(ei.sys.n){};
};

/**
 * @brief      Fixed Input Sample
 */
class ExampleFixedInput : public ExampleInput {
   public:
    using ExampleInput::ExampleInput;
    int assign() override;
};

/**
 * @brief      (1) Each matrix entry randomly generated from a uniform distribution
 *             on [−1,1] (denoted as U(−1,1))
 */
class ExampleRandom : public ExampleInput {
   public:
    using ExampleInput::ExampleInput;
    int assign() override;
};

/**
 * @brief      Random RHS(U(-1, 1)) Input
 */
class ExampleRandomRHSInput : public ExampleInput {
   public:
    using ExampleInput::ExampleInput;
    int assign() override;
};
}  // namespace trisys
