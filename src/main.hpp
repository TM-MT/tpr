#pragma once

#include <vector>

#include "lib.hpp"

#define EPS 1e-3

struct TRIDIAG_SYSTEM {
    int n;
    real *a;
    real *diag;
    real *c;
    real *rhs;
};

int setup(struct TRIDIAG_SYSTEM *sys, int n);
int clean(struct TRIDIAG_SYSTEM *sys);
int assign(struct TRIDIAG_SYSTEM *sys);
bool sys_null_check(struct TRIDIAG_SYSTEM *sys);

class TPR_ANS {
   public:
    int n;
    int s;
    float *x;
    std::vector<float> data;

    TPR_ANS(int n) {
        this->n = n;
        this->data.resize(n);
        this->x = this->data.data();
    }

    ~TPR_ANS() {}

    bool operator==(const TPR_ANS &ans) {
        assert(this->n == ans.n);
        bool ret = true;

        for (int i = 0; i < this->n; i++) {
            ret &= fequal(this->x[i], ans.x[i]);
        }

        return true;
    }

    bool operator!=(const TPR_ANS &ans) { return !(*this == ans); }

    // operator `<<` cannot define for following reason,
    // nvcc 11: Compile Fail
    // `error: too many parameters for this operator function`
    // std::ostream& operator<<(std::ostream &os, TPR_ANS const &ans) {
    //     for (int i = 0; i < n; i++) {
    //         os << ans.x[i] << ", ";
    //     }
    //     return os << "\n";
    // }
    void display(std::ostream &os) {
        for (int i = 0; i < n; i++) {
            os << this->x[i] << ", ";
        }
        os << "\n";
    }

   private:
    bool fequal(float a, float b) {
        return fabs(a - b) <= EPS * fmax(1, fmax(fabs(a), fabs(b)));
    }
};

#undef EPS
