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
};

/**
 * @brief      The base class for sample input
 */
class ExampleInput {
   public:
    struct TRIDIAG_SYSTEM sys;
    ExampleInput(int n) : sys(n){};
    ~ExampleInput(){};
    virtual int assign() { return 0; };
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
 * @brief      Random RHS(U(-1, 1)) Input
 */
class ExampleRandomRHSInput : public ExampleInput {
   public:
    using ExampleInput::ExampleInput;
    int assign() override;
};
}  // namespace trisys
