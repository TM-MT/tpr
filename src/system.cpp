#include "system.hpp"

#include <stdlib.h>
using namespace trisys;

TRIDIAG_SYSTEM::TRIDIAG_SYSTEM(int n) {
    this->a = (real *)malloc(n * sizeof(real));
    this->diag = (real *)malloc(n * sizeof(real));
    this->c = (real *)malloc(n * sizeof(real));
    this->rhs = (real *)malloc(n * sizeof(real));
    this->n = n;

    bool check = null_check();
    if (!check) {
        fprintf(stderr, "Failed at allocation.\n");
        exit(EXIT_FAILURE);
    }
}

bool TRIDIAG_SYSTEM::null_check() {
    for (auto p : {this->a, this->diag, this->c, this->rhs}) {
        if (p == nullptr) {
            return false;
        }
    }
    return true;
}

TRIDIAG_SYSTEM::~TRIDIAG_SYSTEM() {
    for (auto p : {this->a, this->diag, this->c, this->rhs}) {
        free(p);
    }

    this->a = nullptr;
    this->diag = nullptr;
    this->c = nullptr;
    this->rhs = nullptr;
}

int ExampleFixedInput::assign() {
    int n = this->sys.n;
    for (int i = 0; i < n; i++) {
        this->sys.a[i] = -1.0 / 6.0;
        this->sys.c[i] = -1.0 / 6.0;
        this->sys.diag[i] = 1.0;
        this->sys.rhs[i] = 1.0 * (i + 1);
    }
    this->sys.a[0] = 0.0;
    this->sys.c[n - 1] = 0.0;

    return 0;
}
