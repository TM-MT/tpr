#pragma once
#include "lib.hpp"

/**
 * @brief      x = (real *)malloc(sizeof(real) * n)
 *
 * @param      x     *real
 * @param      n     length of array
 *
 */
#define RMALLOC(x, n) x = new real[n]

/**
 * @brief Safely delete pointer `p` and set `p = nullptr`
 */
#define SAFE_DELETE(p) \
    delete[] p;        \
    p = nullptr

class CR : Solver {
    real *a, *c, *rhs;
    real *aa, *cc, *rr;
    real *x;
    int n;

   public:
    CR(real *a, real *diag, real *c, real *rhs, int n) {
        init(n);
        set_tridiagonal_system(a, diag, c, rhs);
    };

    CR(int n) { init(n); }

    CR(){};

    ~CR() {
        SAFE_DELETE(this->aa);
        SAFE_DELETE(this->cc);
        SAFE_DELETE(this->rr);
        SAFE_DELETE(this->x);

#pragma acc exit data detach(this->a, this->c, this->rhs)
#pragma acc exit data delete ( \
    this->aa[:n], this->cc[:n], this->rr[:n], this->x[:n])
#pragma acc exit data delete (this)
    }

    void set_tridiagonal_system(real *a, real *diag, real *c, real *rhs);

    int solve();

    int get_ans(real *x);

    /**
     * @brief Initialize CR with size `n`
     * @note call this before call any function in CR
     *
     * @param n size of the system
     */
    void init(int n) {
        this->n = n;

        RMALLOC(this->aa, this->n);
        RMALLOC(this->cc, this->n);
        RMALLOC(this->rr, this->n);
        RMALLOC(this->x, this->n);

        if ((this->aa == nullptr) || (this->cc == nullptr) ||
            (this->rr == nullptr) || (this->x == nullptr)) {
            abort();
        }

#pragma acc enter data copyin(this)
#pragma acc enter data create(this->aa [0:n], this->cc [0:n], this->rr [0:n], \
                              this->x [0:n])
    }

   private:
    CR(const CR &cr);
    CR &operator=(const CR &cr);

    int fr();
    int bs();
};

#undef RMALLOC
#undef SAFE_DELETE
