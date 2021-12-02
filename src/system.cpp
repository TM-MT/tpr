#include "system.hpp"

#include <assert.h>
#include <stdlib.h>

#include <cstdio>

#include "effolkronium/random.hpp"

// std::mt19937 base pseudo-random
using Random = effolkronium::random_static;

using namespace trisys;

/**
 * @brief      Read double and check
 */
#define READFP(dst) assert(fscanf(fp, "%lf,", &dst) == 1)

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

int ExampleFromInput::assign() {
    std::cerr << "Reading from " << this->path << "\n";
    FILE *fp = fopen(this->path.c_str(), "r");
    int n = this->sys.n;
    bool failure = false;
    double dummy;
    double *tmp_a = new double[n];
    double *tmp_diag = new double[n];
    double *tmp_c = new double[n];
    double *tmp_rhs = new double[n];

    // Read the file with double precision
    if (fp != nullptr) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i - 1; j++) {
                READFP(dummy);
            }

            if (i > 0) {
                READFP(tmp_a[i]);
            } else {
                tmp_a[i] = 0.0;
            }

            READFP(tmp_diag[i]);
            if (i < n - 1) {
                READFP(tmp_c[i]);
            } else {
                tmp_c[i] = 0.0;
            }

            for (int j = i + 2; j < n; j++) {
                READFP(dummy);
            }
            assert(fscanf(fp, "%lf", &tmp_rhs[i]) == 1);
        }
    } else {
        std::cerr << "FAILED at opening " << this->path << "\n";
        failure = true;
    }

    // copy
    if (!failure) {
        for (int i = 0; i < n; i++) {
            this->sys.a[i] = static_cast<real>(tmp_a[i]);
            this->sys.diag[i] = static_cast<real>(tmp_diag[i]);
            this->sys.c[i] = static_cast<real>(tmp_c[i]);
            this->sys.rhs[i] = static_cast<real>(tmp_rhs[i]);
        }
    }

    delete[] tmp_a;
    delete[] tmp_diag;
    delete[] tmp_c;
    delete[] tmp_rhs;

    fclose(fp);

    if (failure) {
        exit(EXIT_FAILURE);
    }
    return 0;
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

int ExampleRandom::assign() {
    int n = this->sys.n;
    for (int i = 0; i < n; i++) {
        this->sys.a[i] = Random::get(-1., 1.);     // U(-1, 1)
        this->sys.c[i] = Random::get(-1., 1.);     // U(-1, 1)
        this->sys.diag[i] = Random::get(-1., 1.);  // U(-1, 1)
        this->sys.rhs[i] = Random::get(-1., 1.);   // U(-1, 1)
    }
    this->sys.a[0] = 0.0;
    this->sys.c[n - 1] = 0.0;

    return 0;
}

int ExampleRandomRHSInput::assign() {
    int n = this->sys.n;
    for (int i = 0; i < n; i++) {
        this->sys.a[i] = -1.0 / 6.0;
        this->sys.c[i] = -1.0 / 6.0;
        this->sys.diag[i] = 1.0;
        this->sys.rhs[i] = Random::get(-1., 1.);  // U(-1, 1)
    }
    this->sys.a[0] = 0.0;
    this->sys.c[n - 1] = 0.0;

    return 0;
}
